from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from typing import Optional
import os
from dotenv import load_dotenv

# ==================================
# Environment Setup
# ==================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")
print(f"🔧 Loading environment from: {env_path}")
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("GPT4_API_KEY")
endpoint = os.getenv("GPT4_ENDPOINT")
print(f"🔑 GPT4_API_KEY loaded: {bool(api_key)}")
print(f"🌐 GPT4_ENDPOINT loaded: {bool(endpoint)}")

# ==================================
# Import Model Logic
# ==================================
from model import rag_agentic_chatbot

# ==================================
# FastAPI App Initialization
# ==================================
app = FastAPI(title="Uttar Pradesh RAG AI Assistant")

# In-memory chat history (can replace with Redis/DB later)
session_histories = {}

# ==================================
# Pydantic Models
# ==================================
class ChatRequest(BaseModel):
    user_query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    user_query: str
    ai_reply: str

# ==================================
# Chat Endpoint
# ==================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for WhatsApp AI Assistant.
    Accepts a user query and optional session_id.
    """
    session_id = request.session_id or str(uuid.uuid4())
    history = session_histories.get(session_id, [])

    ai_reply, updated_history = rag_agentic_chatbot(request.user_query, history)
    session_histories[session_id] = updated_history

    return ChatResponse(
        session_id=session_id,
        user_query=request.user_query,
        ai_reply=ai_reply
    )

# ==================================
# Health Check Endpoint
# ==================================
@app.get("/health")
def health():
    """Simple health check."""
    return {"status": "ok"}

# ==================================
# Diagnostic Endpoint
# ==================================
@app.get("/test_env")
def test_env():
    """
    Diagnostic endpoint to confirm environment variables are loaded properly.
    Returns masked API key and endpoint for debugging.
    """
    api_key = os.getenv("GPT4_API_KEY")
    endpoint = os.getenv("GPT4_ENDPOINT")
    return {
        "api_key_loaded": bool(api_key),
        "endpoint_loaded": bool(endpoint),
        "api_key_start": (api_key[:10] + "...") if api_key else None,
        "endpoint_start": (endpoint[:40] + "...") if endpoint else None,
        "working_directory": os.getcwd(),
        "env_path_used": env_path
    }

# ==================================
# Local Development Entry Point
# ==================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
