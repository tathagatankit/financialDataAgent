import os
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import uuid
import sqlite3
import json
from typing import Literal
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, message_to_dict, messages_from_dict
from langgraph.graph import END, START, MessagesState, StateGraph
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Set up Google API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
llm = model

# Initialize SQLDatabase
db = SQLDatabase.from_uri("sqlite:///stock.db")

# Initialize SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Define nodes for the graph
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")

run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")

# Example: create a predetermined tool call
def list_tables(state: MessagesState):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")

    return {"messages": [tool_call_message, tool_message, response]}

# Example: force a model to create a tool call
def call_get_schema(state: MessagesState):
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

generate_query_system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer.

Use the conversation history to answer follow-up questions. For example, if the user asks for the average of the last 6 closing prices, and the previous turn contains the last 6 closing prices, you should calculate the average from those prices.

The database is about stocks and contains tables such as 'stock_index_price_daily'.
The tables contain historical stock index prices with columns like 'date_key', 'open', 'high', 'low', 'close', and 'index_name'.

When querying for an index name, use the UPPER function to ensure case-insensitive matching. For example, to find 'nifty auto', use `WHERE UPPER(index_name) = 'NIFTY AUTO'`.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
""".format(
    dialect=db.dialect,
    top_k=5,
)

def generate_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }
    llm_with_tools = llm.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}

check_query_system_prompt = """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
""".format(dialect=db.dialect)

def check_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": check_query_system_prompt,
    }
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id
    return {"messages": [response]}

def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "check_query"

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(run_query_node, "run_query")

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    should_continue,
)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

agent = builder.compile()


from datetime import datetime

def get_history(session_id: str) -> list:
    """Fetches conversation history from the database."""
    conn = sqlite3.connect("stock.db")
    cursor = conn.cursor()
    cursor.execute("SELECT messages FROM conversation_history WHERE session_id = ?", (session_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return messages_from_dict(json.loads(result[0]))
    return []

def save_history(session_id: str, messages: list):
    """Saves conversation history to the database."""
    conn = sqlite3.connect("stock.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO conversation_history (session_id, messages, timestamp) VALUES (?, ?, ?)",
        (session_id, json.dumps([message_to_dict(m) for m in messages]), datetime.now()),
    )
    conn.commit()
    conn.close()


# FastAPI application
app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


@app.post("/query")
async def run_agent_query(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())

    # Retrieve conversation history or start a new one
    history = get_history(session_id)

    # Add the new user message
    messages = history + [HumanMessage(content=request.query)]

    # Invoke the agent with the conversation history
    final_state = None
    for s in agent.stream({"messages": messages}):
        final_state = s

    if not final_state:
        return {"response": "Agent failed to produce a response.", "session_id": session_id}

    # The final state is a dictionary where the last value is the most recent state
    final_messages = list(final_state.values())[-1]["messages"]

    # Update the history for the session
    save_history(session_id, final_messages)

    # Find the last AIMessage to send back to the user
    final_answer = "No response."
    for message in reversed(final_messages):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            final_answer = message.content
            break

    return {"response": final_answer, "session_id": session_id}

@app.get("/")
async def root():
    return {"message": "Stock Database Query Agent API. Use /query endpoint to send queries."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
