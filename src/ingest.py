from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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


# ─────────────────────────────────────────────
# STEP 1: LOAD DOCUMENTS
# ─────────────────────────────────────────────
def load_documents():
    print("📂 Loading documents...")
    loader = DirectoryLoader(
        "../documents",
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents")
    for doc in documents:
        print(f"   → {doc.metadata['source']}")
    return documents


# ─────────────────────────────────────────────
# STEP 2: CREATE PARENT-CHILD CHUNKS
# ─────────────────────────────────────────────
def create_parent_child_chunks(documents):
    """
    Parent = large chunks (full product sections)
             Used for: sending to LLM as context
    
    Child  = small chunks (specific facts)
             Used for: precise Pinecone search
    """
    print("\n✂️  Creating parent-child chunks...")

    # Parent splitter — large, captures full sections
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,     # large — full product section
        chunk_overlap=150,   # generous overlap
        separators=["\n\n", "\n", " ", ""]
    )

    # Child splitter — small, captures specific facts
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,      # small — specific facts
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    all_pairs = []
    parent_count = 0
    child_count = 0

    for doc in documents:
        source = doc.metadata['source']

        # Create parent chunks from document
        parents = parent_splitter.split_documents([doc])

        for parent in parents:
            parent_id = f"parent_{parent_count}"

            # Create child chunks FROM this parent
            children = child_splitter.split_documents([parent])

            for child in children:
                pair = {
                    "child_id":    f"child_{child_count}",
                    "child_text":  child.page_content,
                    "parent_id":   parent_id,
                    "parent_text": parent.page_content,
                    "source":      source
                }
                all_pairs.append(pair)
                child_count += 1

            parent_count += 1

    print(f"✅ Created {parent_count} parents → {child_count} children")

    # Show a sample so we can verify!
    print("\n--- Sample Parent (what LLM reads) ---")
    print(f"Length: {len(all_pairs[0]['parent_text'])} chars")
    print(all_pairs[0]['parent_text'][:300] + "...")

    print("\n--- Sample Child (what we search) ---")
    print(f"Length: {len(all_pairs[0]['child_text'])} chars")
    print(all_pairs[0]['child_text'][:150] + "...")

    return all_pairs


# ─────────────────────────────────────────────
# STEP 3: STORE IN PINECONE
# ─────────────────────────────────────────────
def store_in_pinecone(pairs):
    """
    Store CHILD embedding as vector
    Store PARENT text in metadata
    
    Search = child precision
    Answer = parent context
    """
    print(f"\n🌲 Storing {len(pairs)} chunks in Pinecone...")

    # Clear existing vectors first!
    print("🗑️  Clearing existing Pinecone data...")
    index.delete(delete_all=True)
    time.sleep(2)  # wait for deletion to complete
    print("✅ Pinecone cleared!")

    stored = 0
    for i, pair in enumerate(pairs):

        # Embed the CHILD text (small, precise)
        child_embedding = embedding_model.embed_query(
            pair["child_text"]
        )

        # Store with PARENT text in metadata!
        index.upsert(vectors=[{
            "id": pair["child_id"],
            "values": child_embedding,      # child vector
            "metadata": {
                # For searching context display
                "child_text":  pair["child_text"],

                # For LLM context — THE KEY PART! 🔑
                "parent_text": pair["parent_text"],
                "parent_id":   pair["parent_id"],

                # For reference
                "source":      pair["source"]
            }
        }])

        stored += 1
        print(f"   ✅ {pair['child_id']} → {pair['parent_id']} "
              f"({pair['source'].split(chr(92))[-1]})")

        # Rate limiting protection
        time.sleep(0.5)

    print(f"\n🎉 Successfully stored {stored} chunks!")
    print(f"   Each with parent context for better answers!")


# ─────────────────────────────────────────────
# STEP 4: VERIFY
# ─────────────────────────────────────────────
def verify_pinecone():
    """Quick verification that data is stored"""
    print("\n🔍 Verifying Pinecone storage...")
    time.sleep(2)  # wait for indexing

    stats = index.describe_index_stats()
    print(f"✅ Total vectors in Pinecone: "
          f"{stats['total_vector_count']}")


# ─────────────────────────────────────────────
# RUN EVERYTHING
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting Parent-Child Ingestion Pipeline")
    print("=" * 50)

    documents = load_documents()
    pairs = create_parent_child_chunks(documents)
    store_in_pinecone(pairs)
    verify_pinecone()

    print("\n✅ Ingestion Complete!")
    print(f"   → Documents loaded:  4")
    print(f"   → Parent chunks:     check output above")
    print(f"   → Child chunks:      check output above")
    print(f"   → Ready for search!  🎯")