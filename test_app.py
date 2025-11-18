import requests
import uuid
import time

BASE_URL = "http://0.0.0.0:8000"

def test_conversation_flow():
    session_id = str(uuid.uuid4())
    
    # Question 1
    response = requests.post(
        f"{BASE_URL}/query",
        json={"query": "what is the last 6 closing price of Nifty Bank", "session_id": session_id},
    )
    assert response.status_code == 200
    assert "closing prices" in response.json()["response"]
    time.sleep(15)

    # Question 2
    response = requests.post(
        f"{BASE_URL}/query",
        json={"query": "what is their average?", "session_id": session_id},
    )
    assert response.status_code == 200
    assert "average" in response.json()["response"]
    time.sleep(15)

    # Question 3
    response = requests.post(
        f"{BASE_URL}/query",
        json={"query": "what was the highest closing price among them?", "session_id": session_id},
    )
    assert response.status_code == 200
    assert "highest closing price" in response.json()["response"]
    time.sleep(15)

    # Question 4
    response = requests.post(
        f"{BASE_URL}/query",
        json={"query": "what was the lowest closing price among them?", "session_id": session_id},
    )
    assert response.status_code == 200
    assert "lowest closing price" in response.json()["response"]
    time.sleep(15)

    # Question 5
    response = requests.post(
        f"{BASE_URL}/query",
        json={"query": "what is the difference between the highest and lowest closing price?", "session_id": session_id},
    )
    assert response.status_code == 200
    assert "difference" in response.json()["response"]
