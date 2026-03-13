from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langsmith import traceable
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import time

load_dotenv()

# ─────────────────────────────────────────────
# CONNECTIONS
# ─────────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# ─────────────────────────────────────────────
# MEMORY STORE
# Simple dict → session_id: [messages]
# Like HttpSession in Spring Boot!
# ─────────────────────────────────────────────
conversation_histories: dict = {}
MAX_HISTORY = 10  # last 10 exchanges = 20 messages

def get_history(session_id: str) -> list:
    """Get or create history for session"""
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
        print(f"🆕 New session: {session_id}")
    return conversation_histories[session_id]

def save_to_history(session_id: str, question: str, answer: str):
    """Save exchange to history with window trimming"""
    history = get_history(session_id)

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    # Trim to window size — like a circular buffer!
    max_messages = MAX_HISTORY * 2
    if len(history) > max_messages:
        conversation_histories[session_id] = history[-max_messages:]
        print(f"✂️ Trimmed to last {MAX_HISTORY} exchanges")

def clear_memory(session_id: str):
    """Clear session — called when user starts new chat"""
    if session_id in conversation_histories:
        del conversation_histories[session_id]
        print(f"🗑️ Cleared session: {session_id}")


# ─────────────────────────────────────────────
# SECURITY
# ─────────────────────────────────────────────
@traceable(name="prompt-injection-check")
def detect_prompt_injection(question: str) -> bool:
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
            print(f"⚠️ Injection detected: '{pattern}'")
            return True
    return False


@traceable(name="query-enhancer")
def enhance_query(question: str, history: list) -> str:
    """
    Use LLM to rewrite vague queries into specific ones
    before searching Pinecone!
    
    "tell me about second one" + history
    → "Tell me about FirstBank Gold Card"
    """
    # Only enhance if there's history — otherwise use as-is
    if not history:
        return question

    # Build recent history string (last 4 messages only)
    recent = history[-4:]
    history_text = ""
    for msg in recent:
        if isinstance(msg, HumanMessage):
            history_text += f"Customer: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Surabhi: {msg.content[:200]}...\n"

    enhancement_prompt = f"""Given this conversation history:
{history_text}

The customer now asks: "{question}"

If the question contains vague references like "second one", 
"that card", "it", "the first one", "tell me more" etc.,
rewrite it as a specific standalone search query.

If the question is already specific, return it unchanged.

Return ONLY the rewritten query, nothing else.
No explanation, no quotes, just the query."""

    try:
        response = llm.invoke([HumanMessage(content=enhancement_prompt)])
        enhanced = response.content.strip()
        
        if enhanced != question:
            print(f"🔄 Query enhanced:")
            print(f"   Original: '{question}'")
            print(f"   Enhanced: '{enhanced}'")
        
        return enhanced
    except Exception:
        # If enhancement fails, use original query
        return question


# ─────────────────────────────────────────────
# SEARCH & CONTEXT
# ─────────────────────────────────────────────
@traceable(name="pinecone-search")
def search_documents(query: str, top_k: int = 3) -> list:
    """Search using child embeddings"""
    query_embedding = embedding_model.embed_query(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results['matches']

@traceable(name="context-builder")
def build_context(matches: list) -> str:
    """
    Return PARENT text to LLM — not child text!
    Deduplicate parents so same section not sent twice!
    """
    context = ""
    seen_parents = set()  # deduplication!

    for match in matches:
        parent_id = match['metadata'].get('parent_id', '')
        parent_text = match['metadata'].get('parent_text', 
                      match['metadata'].get('text', ''))

        # Skip if we already added this parent!
        if parent_id in seen_parents:
            print(f"   ⏭️  Skipping duplicate parent: {parent_id}")
            continue

        seen_parents.add(parent_id)
        source = match['metadata']['source'].split('\\')[-1]

        context += f"\n--- Source: {source} ---\n"
        context += parent_text
        context += "\n"

        print(f"   📄 Using {parent_id} from {source} "
              f"(score: {match['score']:.4f})")

    return context
# ─────────────────────────────────────────────
# MAIN RAG PIPELINE WITH MEMORY
# ─────────────────────────────────────────────
@traceable(name="financial-rag-pipeline")
def ask_financial_chatbot(question: str, session_id: str = "default") -> str:

    if detect_prompt_injection(question):
        return ("I'm sorry, I cannot process that request. "
                "Please call 1-800-FIRST-BK for assistance.")

    history = get_history(session_id)

    print(f"\n👤 Customer [{session_id}]: {question}")
    print(f"📝 History: {len(history)} messages in memory")
    print("─" * 50)

    # 🔄 Enhance vague queries BEFORE searching Pinecone!
    search_query = enhance_query(question, history)

    # Search with ENHANCED query!
    print("🔍 Searching knowledge base...")
    matches = search_documents(search_query)   # ← enhanced!
    context = build_context(matches)

    # Build messages with history
    messages = []
    messages.append(SystemMessage(content=f"""You are Surabhi, a helpful \
and professional customer service assistant for FirstBank.

IMPORTANT RULES:
1. Answer using ONLY the context provided below
2. If answer not in context, say you don't have that information
   and suggest calling 1-800-FIRST-BK
3. Use conversation history to understand follow-up questions
4. Always be professional and friendly
5. Never make up information

Context from FirstBank knowledge base:
{context}"""))

    messages.extend(history)
    messages.append(HumanMessage(content=question))  # original question!

    # Generate with retry
    max_retries = 3
    retry_delay = 10

    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)

            if isinstance(response.content, list):
                answer = response.content[0]['text']
            else:
                answer = response.content

            save_to_history(session_id, question, answer)

            print(f"\n🏦 Surabhi: {answer[:100]}...")
            return answer

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    print(f"⚠️ Rate limited. Retry in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return ("I'm experiencing high demand. "
                           "Please try again or call 1-800-FIRST-BK.")
            else:
                print(f"❌ Error: {e}")
                return ("I apologize, something went wrong. "
                       "Please call 1-800-FIRST-BK.")

    return "Service unavailable. Please call 1-800-FIRST-BK."
