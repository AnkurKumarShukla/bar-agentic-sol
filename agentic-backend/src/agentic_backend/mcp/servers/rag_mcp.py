from mcp.server.fastmcp import FastMCP
import os
import sys
import glob
import chromadb
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key
client = OpenAI()
mcp = FastMCP("RAG Search MCP Server")
# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="../chroma_db")

def get_embedding(text):
    """Get OpenAI embedding for given text."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# -------------------- Collection Setup --------------------

def setup_collection(collection_name="documents"):
    """Create or retrieve a ChromaDB collection."""
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}", file=sys.stderr, flush=True)
    except:
        collection = chroma_client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}", file=sys.stderr, flush=True)
    return collection

# -------------------- Add Documents --------------------

def add_documents(documents, collection_name="documents"):
    """
    Add documents to ChromaDB collection.
    Each document must be a dict with 'text' and optional 'metadata'.
    """
    collection = setup_collection(collection_name)
    
    texts = []
    embeddings = []
    metadatas = []
    ids = []
    
    for i, doc in enumerate(documents):
        text = doc['text']
        metadata = doc.get('metadata', {})
        
        embedding = get_embedding(text)
        
        texts.append(text)
        embeddings.append(embedding)
        metadatas.append(metadata)
        ids.append(f"doc_{i}")
    
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Added {len(documents)} documents to collection '{collection_name}'.", file=sys.stderr, flush=True)
    return collection

# -------------------- Semantic Search --------------------

def semantic_search(query, top_k=5, collection_name="documents"):
    """Perform semantic search against the ChromaDB collection."""
    try:
        collection = chroma_client.get_collection(name=collection_name)
        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]
                })
        return formatted_results
    except Exception as e:
        print(f"Error during semantic search: {e}", file=sys.stderr, flush=True)
        return []

def query_documents(query, top_k=5, collection_name="documents"):
    """Query documents and print results."""
    results = semantic_search(query, top_k, collection_name)

    if not results:
        print("No results found.", file=sys.stderr, flush=True)
        return []

    print(f"\nFound {len(results)} results for query: '{query}'", file=sys.stderr, flush=True)
    print("-" * 50, file=sys.stderr, flush=True)

    for i, result in enumerate(results):
        print(f"Result {i+1}:", file=sys.stderr, flush=True)
        print(f"Similarity: {result['similarity']:.4f}", file=sys.stderr, flush=True)
        print(f"Content: {result['content'][:300]}...", file=sys.stderr, flush=True)
        print(f"Metadata: {result['metadata']}", file=sys.stderr, flush=True)
        print("-" * 30, file=sys.stderr, flush=True)

    return results

@mcp.tool()
def search(query:str)->dict:
    """ this tool help to fetch result from docs . currently have ESG POLICY . this will fetch result from docs give by user """
    return query_documents(query,5, "documents")

if __name__ == "__main__":
    mcp.run()