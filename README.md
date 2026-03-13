# 🏦 FirstBank AI Financial Chatbot

An enterprise-grade **Agentic AI chatbot** for financial services, 
built with RAG (Retrieval Augmented Generation) architecture.

## 🎯 Features

- **RAG Pipeline** — Answers grounded in real bank documents
- **Semantic Search** — Pinecone vector DB with 3072-dim embeddings
- **Prompt Injection Detection** — Security against AI manipulation
- **LangSmith Observability** — Full request tracing & monitoring
- **Graceful Error Handling** — Production-grade error management
- **Angular 19 UI** — Standalone components with Signals

## 🏗️ Architecture
```
Angular 19 Frontend (Signals + Standalone)
          ↓ HTTP POST /api/chat
FastAPI Backend (Python)
          ↓
LangChain Agent
          ↓
┌─────────────────────────────────┐
│  Pinecone Vector DB             │
│  (gemini-embedding-001, 3072d)  │
└─────────────────────────────────┘
          ↓ Retrieved context
Google Gemini 2.5 Flash (LLM)
          ↓
LangSmith (Observability)
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Angular 19, Signals, Standalone Components |
| Backend | Python, FastAPI, Uvicorn |
| AI Orchestration | LangChain |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google gemini-embedding-001 (3072 dims) |
| Vector DB | Pinecone (Cosine similarity) |
| Observability | LangSmith |

## 🚀 Getting Started

### Backend Setup
```bash
cd financial-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` file:
```
GEMINI_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=financial-chatbot
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=financial-chatbot
```
```bash
# Ingest documents into Pinecone
python src/ingest.py

# Start FastAPI server
uvicorn src/main:app --reload
```

### Frontend Setup
```bash
cd financial-chatbot-ui
npm install
ng serve
```

Open **http://localhost:4200**

## 🔐 Security Features

- Prompt injection detection
- CORS protection
- API key management via environment variables
- Graceful error handling with user-friendly messages
- PII-aware design

## 📊 Supported Queries

- 💳 Credit card offerings & benefits
- 🏠 Loan eligibility & rates  
- 💰 Account services & rates
- ❓ General banking FAQ

## 🏢 Enterprise Considerations

- LangSmith tracing for full observability
- JWT authentication ready (FastAPI middleware)
- Horizontal scaling with Kubernetes
- PII masking for compliance (SOX, PCI-DSS)
- Hybrid search upgrade path (dense + sparse)
- Re-ranking pipeline (Cohere integration ready)