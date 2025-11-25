from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import httpx
import json
import uuid
from typing import Optional
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).parent.resolve()

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

API_URL = "http://127.0.0.1:8000/query"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/send_query", response_class=HTMLResponse)
async def send_query(request: Request, query: str = Form(...), session_id: Optional[str] = Form(None)):
    if not session_id:
        session_id = str(uuid.uuid4())

    async def stream_from_api():
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", API_URL, json={"query": query, "session_id": session_id}, timeout=None) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream_from_api(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
