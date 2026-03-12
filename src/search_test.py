from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

def search_pinecone(query, top_k=3):
    print(f"\n🔍 Searching for: '{query}'")
    print("─" * 50)
    
    # Connect to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    
    # Convert question to embedding
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    query_embedding = embedding_model.embed_query(query)
    print(f"✅ Query converted to {len(query_embedding)} dimensional embedding")
    
    # Search Pinecone for similar vectors
    results = index.query(
        vector=query_embedding,
        top_k=top_k,              # return top 3 most similar chunks
        include_metadata=True      # include original text
    )
    
    # Display results
    print(f"\n📄 Top {top_k} most relevant chunks found:\n")
    for i, match in enumerate(results['matches']):
        print(f"Result {i+1}:")
        print(f"  Score  : {match['score']:.4f}")   # similarity score
        print(f"  Source : {match['metadata']['source']}")
        print(f"  Text   : {match['metadata']['text'][:150]}...")
        print()
    
    return results

# Test with different questions!
search_pinecone("What credit cards do you offer?")
search_pinecone("How do I get a home loan?")
search_pinecone("What is the ATM withdrawal limit?")