# Stock Database Query Agent

This application provides a FastAPI interface to query a stock database using a Langchain agent.

## Project Structure

- `app.py`: The main FastAPI application that exposes the `/query` endpoint.
- `create_db.py`: A script to create the `conversation_history` table in the `stock.db` database.
- `insert_data.py`: A script to insert stock data from CSV files into the `stock.db` database.
- `test_app.py`: A suite of unit tests for the FastAPI application.
- `stock.db`: The SQLite database containing the stock data and conversation history.
- `db.ipynb`: A Jupyter notebook for exploring the database.

## Setup Instructions

Follow these steps to set up and run the application:

### 1. Clone the Repository (if not already cloned)

```bash
git clone https://github.com/ArthamXTradebook/AI.git
cd stock_agent_own
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

- **On macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **On Windows:**
  ```bash
  .\.venv\Scripts\activate
  ```

### 4. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 5. Set up Google API Key

The application requires a Google API Key. Create a `.env` file in the root directory of the project and add your API key:

```
GOOGLE_API_KEY="your_google_api_key_here"
```

Replace `"your_google_api_key_here"` with your actual Google API Key.

### 6. Prepare the Database

First, create the necessary tables in the database by running the `create_db.py` script:

```bash
python create_db.py
```

Then, insert the stock data from the CSV files in the `Market Data` directory by running the `insert_data.py` script:

```bash
python insert_data.py
```

### 7. Run the FastAPI Application

Once all dependencies are installed and the API key is set, you can run the FastAPI application:

```bash
python app.py
```

This will start the server on `http://0.0.0.0:8000`.

### 8. Run the Unit Tests

To run the unit tests, use the following command:

```bash
pytest
```

## API Usage

You can send queries to the `/query` endpoint using a tool like `curl`.

### Example Query

```bash
curl -X 'POST' \
  'http://0.0.0.0:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "what is the last 6 closing price of Nifty Bank",
  "session_id": "your_session_id"
}'
