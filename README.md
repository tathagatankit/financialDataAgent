# Financial Data Agent

A powerful multi-agent system that provides both API-based and web UI interfaces for querying financial data, built with FastAPI and LangChain.

## UI Screenshots

### Welcome Screen
![UI Image 1](img/UI%20Image%201.png)

### Query Interface
![UI Image 2](img/UI%20Image%202.png)

### Response View
![UI Image 3](img/UI%20Image%203.png)

### Advanced Features
![UI Image 4](img/UI%20Image%204.png)

## Project Structure

- `app.py`: The main FastAPI application that exposes the `/query` endpoint.
- `multi_agent_app.py`: Enhanced multi-agent application with streaming capabilities.
- `create_db.py`: A script to create the `conversation_history` table in the `stock.db` database.
- `insert_data.py`: A script to insert stock data from CSV files into the `stock.db` database.
- `test_app.py`: A suite of unit tests for the FastAPI application.
- `stock.db`: The SQLite database containing the stock data and conversation history.
- `db.ipynb`: A Jupyter notebook for exploring the database.
- `ui_interface/`: Contains the modern user interface for the Stock Agent.
  - `ui_app.py`: FastAPI application serving the web interface.
  - `templates/index.html`: The main HTML page for the chat interface.
  - `static/`: Contains CSS, JavaScript, and other static assets.

## Setup Instructions

### 1. Clone the Repository and Set Up Environment

```bash
git clone https://github.com/ArthamXTradebook/AI.git
cd stock_agent_own
python -m venv .venv
source .venv/bin/activate
# On Windows, use: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up Google API Key

Create a `.env` file in the root directory of the project and add your API key:

```
GOOGLE_API_KEY="your_google_api_key_here"
```

### 3. Prepare the Database

```bash
python create_db.py
python insert_data.py
```

### 4. Running the Application

You can run the backend API by itself or with the web UI.

**To Run the Backend API Only:**
```bash
# Basic API
python app.py

# Or the enhanced multi-agent API
uvicorn multi_agent_app:app --host 0.0.0.0 --port 8000
```
The API will be available at `http://0.0.0.0:8000`.

**To Run with the Web UI:**

First, start the backend API in one terminal:
```bash
# Terminal 1: Start Backend API
uvicorn multi_agent_app:app --host 0.0.0.0 --port 8000
```

Then, install UI dependencies and start the UI application in a second terminal:
```bash
# Terminal 2: Start UI Application
cd ui_interface
pip install -r requirements_ui.txt
uvicorn ui_app:app --host 0.0.0.0 --port 8001
```
You can now access the UI in your web browser at `http://127.0.0.1:8001`.

## Usage

### API Usage Example

You can send queries to the `/query` endpoint using a tool like `curl`.

```bash
curl -X 'POST' \
  'http://0.0.0.0:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "what is the last 6 closing price of Nifty Bank",
  "session_id": "your_session_id"
}'
```

### Web UI Usage

1.  Navigate to `http://127.0.0.1:8001` in your browser.
2.  Type your query into the chat interface.
3.  The agent will stream the response back in real-time.

## UI Architecture

The UI is built with the following components:

*   **FastAPI (`ui_app.py`):** Serves the main HTML page and provides a `/send_query` endpoint that streams the response from the backend API.
*   **HTMX:** Handles form submissions and updates the UI with the streaming response without requiring a full page reload.
*   **JavaScript (`app.js`):** Manages the streaming connection, parses the incoming JSON data, and updates the chat window and agent state display in real-time.
*   **CSS (`style.css`):** Provides a modern, responsive design for the chat interface.
