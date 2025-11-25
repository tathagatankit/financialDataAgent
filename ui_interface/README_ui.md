# Stock Agent UI

This directory contains a modern user interface for the Stock Agent, built with FastAPI and HTMX. It provides a real-time, streaming chat experience for interacting with the agent.

## Setup

1.  **Install Dependencies:**
    
    First, ensure you have the necessary Python packages installed. From the `ui_interface` directory, run:
    
    ```bash
    pip install -r requirements_ui.txt
    ```
    
2.  **Run the Backend API:**
    
    Before starting the UI, make sure the main agent API is running. From the root project directory, run:
    
    ```bash
    uvicorn multi_agent_app:app --host 0.0.0.0 --port 8000
    ```
    
3.  **Run the UI Application:**
    
    Once the backend API is running, you can start the UI application. From the `ui_interface` directory, run:
    
    ```bash
    uvicorn ui_app:app --host 0.0.0.0 --port 8001
    ```
    
4.  **Access the UI:**
    
    Open your web browser and navigate to `http://127.0.0.1:8001`. You should see the chat interface, ready to accept your queries.
    

## How It Works

The UI is built with the following components:

*   **FastAPI (`ui_app.py`):** Serves the main HTML page and provides a `/send_query` endpoint that streams the response from the backend API.
*   **HTMX:** Handles form submissions and updates the UI with the streaming response without requiring a full page reload.
*   **JavaScript (`app.js`):** Manages the streaming connection, parses the incoming JSON data, and updates the chat window and agent state display in real-time.
*   **CSS (`style.css`):** Provides a modern, responsive design for the chat interface.

This setup provides a robust and scalable foundation for interacting with your streaming agent, offering a seamless and interactive user experience.
