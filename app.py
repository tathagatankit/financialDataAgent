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
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from typing import Annotated, List
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

# Set up Google API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
model = "gemini-2.5-pro"
llm = ChatGoogleGenerativeAI(model=model)

# Initialize SQLDatabase
db = SQLDatabase.from_uri("sqlite:///stock.db")

# Initialize SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")

stdev_instruction = ""
if db.dialect == "sqlite":
    stdev_instruction = """
To calculate the standard deviation for a column `x`, you can use the formula `SQRT(AVG(x*x) - AVG(x)*AVG(x))`.
This is because the connected database is SQLite, which does not have a built-in standard deviation function.
For example, to calculate the standard deviation of returns, you would first need to calculate the daily returns (e.g., `(close - open) / open`) and then apply the formula.
"""

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

{stdev_instruction}
""".format(
    dialect=db.dialect,
    top_k=5,
    stdev_instruction=stdev_instruction
)

def generate_query(state: AgentState):
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

def check_query(state: AgentState):
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

# SQL Generation Agent
def sql_generator_agent(state: AgentState):
    """
    Generates a SQL query based on the user's question.
    """
    return generate_query(state)

# Query Validation Agent
def query_validation_agent(state: AgentState):
    """
    Validates the SQL query for correctness and security.
    """
    return check_query(state)

data_analyst_system_prompt = """
You are a data analyst agent. Your task is to analyze the results of a SQL query
and provide a concise, insightful summary. The user's original question will be
provided for context. Do not just repeat the data; interpret it and highlight
the key findings.
"""

# Data Analyst Agent
def data_analyst_agent(state: AgentState):
    """
    Analyzes the data and provides insights.
    """
    user_query = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            user_query = m.content
            break
    
    sql_result = state["messages"][-1].content

    prompt = f"Original question: {user_query}\n\nSQL query result:\n{sql_result}"
    
    system_message = {
        "role": "system",
        "content": data_analyst_system_prompt,
    }
    
    response = llm.invoke([system_message, HumanMessage(content=prompt)])
    return {"messages": [response]}

response_generator_system_prompt = """
You are a response generation agent. Your task is to take the analysis from the
data analyst and formulate a clear, user-friendly final answer. Ensure the
response is easy to understand and directly answers the user's original question.
"""

# Response Generation Agent
def response_generator_agent(state: AgentState):
    """
    Generates a final, human-readable response.
    """
    user_query = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            user_query = m.content
            break
            
    analyst_summary = state["messages"][-1].content

    prompt = f"Original question: {user_query}\n\nAnalysis summary:\n{analyst_summary}"

    system_message = {
        "role": "system",
        "content": response_generator_system_prompt,
    }

    response = llm.invoke([system_message, HumanMessage(content=prompt)])
    return {"messages": [response]}

def should_validate_query(state: AgentState) -> Literal["query_validator", "response_generator"]:
    """
    Determines whether to validate the query or generate a response.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        # If there are no tool calls, the LLM might have responded directly.
        return "response_generator"
    else:
        # If there are tool calls, proceed to validation.
        return "query_validator"

# Define the graph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("sql_generator", sql_generator_agent)
builder.add_node("query_validator", query_validation_agent)
builder.add_node("data_analyst", data_analyst_agent)
builder.add_node("response_generator", response_generator_agent)
run_query_node = ToolNode([run_query_tool], name="run_query")
builder.add_node("run_query", run_query_node)

# Define edges
builder.add_edge(START, "sql_generator")
builder.add_conditional_edges(
    "sql_generator",
    should_validate_query,
    {
        "query_validator": "query_validator",
        "response_generator": "response_generator",
    },
)
builder.add_edge("query_validator", "run_query")
builder.add_edge("run_query", "data_analyst")
builder.add_edge("data_analyst", "response_generator")
builder.add_edge("response_generator", END)

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
