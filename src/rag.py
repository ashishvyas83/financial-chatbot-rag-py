from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage           # ← ADD THIS
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import langsmith

load_dotenv()

# Initialize all connections
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

def detect_prompt_injection(question: str) -> bool:
    """
    Detect if user is trying to manipulate the AI
    Critical security feature for banking applications!
    """
    # Common prompt injection patterns
    injection_patterns = [
        "ignore previous instructions",
        "ignore all instructions",
        "forget you are",
        "you are now",
        "pretend you are",
        "act as",
        "jailbreak",
        "bypass",
        "override instructions",
        "disregard",
        "new instruction",
        "system prompt",
    ]
    question_lower = question.lower()
    
    for pattern in injection_patterns:
        if pattern in question_lower:
            print(f"⚠️ Detected prompt injection '{pattern}'")
            return True

    return False


def search_documents(query, top_k=3):
    """Search Pinecone for relevant chunks"""
    query_embedding = embedding_model.embed_query(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results['matches']

def build_context(matches):
    """Combine retrieved chunks into context string"""
    context = ""
    for i, match in enumerate(matches):
        context += f"\n--- Document {i+1} ---\n"
        context += match['metadata']['text']
        context += "\n"
    return context

def ask_financial_chatbot(question):
    """Full RAG pipeline — search + generate"""
    
     # 🛡️ Security check FIRST — before anything else!
    if detect_prompt_injection(question):
        return ("I'm sorry, I cannot process that request. "
                "For assistance please call 1-800-FIRST-BK "
                "or visit your nearest FirstBank branch.")
        
    print(f"\n👤 Customer: {question}")
    print("─" * 50)
    
    # Step 1: Search Pinecone for relevant chunks
    print("🔍 Searching knowledge base...")
    matches = search_documents(question)
    
    # Step 2: Build context from retrieved chunks
    context = build_context(matches)
    
    # Step 3: Build prompt with context
    prompt = f"""You are Surabhi, a helpful and professional customer service 
assistant for FirstBank. Answer the customer's question using ONLY 
the information provided in the context below.

If the answer is not in the context, politely say you don't have 
that information and suggest they call 1-800-FIRST-BK.

Context from our knowledge base:
{context}

Customer Question: {question}

Provide a clear, helpful, and professional response:"""

    # Step 4: Send to Gemini via LangChain
    print("🤖 Generating response...")
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Handle both string and list responses
    if isinstance(response.content, list):
        answer = response.content[0]['text']  # ← extract from list
    else:
        answer = response.content             # ← already a string
    
    print(f"\n🏦 Surabhi: {answer[:100]}...")
    print("─" * 50)
    
    return answer

# Test it with real financial questions!
#ask_financial_chatbot("What credit cards do you offer?")
#ask_financial_chatbot("How do I apply for a home loan?")
#ask_financial_chatbot("What is the ATM withdrawal limit?")
#ask_financial_chatbot("What is the weather like today?")  # out of scope test!