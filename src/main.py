from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ask_financial_chatbot, clear_memory
from dotenv import load_dotenv
import os

load_dotenv()

print(f"🔍 LangSmith tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"📊 Project: {os.getenv('LANGCHAIN_PROJECT')}")

app = FastAPI(
    title="FirstBank AI Chatbot API",
    description="RAG-powered financial assistant with memory",
    version="2.0.0"          # ← upgraded to v2!
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    session_id: str

@app.get("/health")
def health_check():
    return {
        "status": "UP",
        "service": "FirstBank AI Chatbot",
        "version": "2.0.0",
        "features": ["RAG", "Memory", "LangSmith", "Security"]
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    print(f"\n📨 [{request.session_id}] {request.question}")
    answer = ask_financial_chatbot(
        request.question,
        request.session_id      # ← pass session ID!
    )
    return ChatResponse(
        answer=answer,
        session_id=request.session_id
    )

# New endpoint — clear conversation memory
@app.delete("/api/chat/{session_id}")
def clear_chat(session_id: str):
    clear_memory(session_id)
    return {"message": f"Session {session_id} cleared!"}

@app.get("/")
def root():
    return {"message": "FirstBank AI Chatbot v2.0 — Now with Memory!"}