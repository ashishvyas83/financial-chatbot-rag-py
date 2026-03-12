from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import time

load_dotenv()

def load_documents():
    print("Loading documents...")


    loader = DirectoryLoader(
        path="../documents",        # folder with our 4 txt files
        glob="*.txt",               # only load .txt files
        loader_cls=TextLoader       # use TextLoader for each file
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents")

    for doc in documents:
        print(f" -> {doc.metadata['source']}")

    return documents


def chunk_documents(documents):

    print("\nChunking documents...")

    # 1. VALIDATE before chunking
    cleaned_documents = []
    for doc in documents:
        content = doc.page_content
        
        # Check if document looks readable
        if len(content) == 0:
            print(f"⚠️ Skipping empty document: {doc.metadata['source']}")
            continue
            
        # Check if it has any spaces (basic readability check)
        space_ratio = content.count(' ') / len(content)
        if space_ratio < 0.05:   # less than 5% spaces = suspicious!
            print(f"⚠️ Warning: {doc.metadata['source']} looks malformed")
            # Still process it but warn the user
            
        cleaned_documents.append(doc)
    
    # 2. THEN chunk normally
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]  # "" = last resort force cut
    )
    
    chunks = splitter.split_documents(cleaned_documents)

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    print("\n--- Sample Chunk ---")
    print(f"Content: {chunks[0].page_content}")
    print(f"Source:  {chunks[0].metadata['source']}")
    print(f"Length:  {len(chunks[0].page_content)} characters")

    return chunks


def store_chunks_in_pinecone(chunks):
    pc=Pinecone(appi_key=os.getenv("PINECONE_API_KEY"))
    index=pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    print("Generating embeddings and storing in Pinecone...")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    stored=0

    for i, chunk in enumerate(chunks):
        embedding = embedding_model.embed_query(chunk.page_content)

        index.upsert(vectors=[{
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk.page_content,
                "source": chunk.metadata["source"]
            }
        }])

        stored += 1

        print(f"  Stored chunk {i+1}/{len(chunks)} from {chunk.metadata['source']}")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    print(f"\n✅ Successfully stored {stored} chunks in Pinecone!")
    return index


documents = load_documents()
chunks = chunk_documents(documents)
index = store_chunks_in_pinecone(chunks)
