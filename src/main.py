from rag import ask_financial_chatbot
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

print(f"Langchain tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"Langchain project: {os.getenv('LANGCHAIN_PROJECT')}")


app = FastAPI(
    title="FirstBank RAG API",
    description="API for the FirstBank RAG system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    session_id: str="default"


class ChatResponse(BaseModel):
    answer: str
    session_id: str

@app.get("/health")
def health_check():
    return {
        "status": "UP",
        "service": "FirstBank RAG API",
        "version": "1.0.0",
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    print(f"\n📨 Received question: {request.question}")
    
    # Call our RAG pipeline
    answer = ask_financial_chatbot(request.question)
    
    return ChatResponse(
        answer=answer,
        session_id=request.session_id
    )


@app.get("/")
def root():
    return {"message": "Welcome to the FirstBank RAG API"}


