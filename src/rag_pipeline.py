"""
RAG Pipeline Module
Handles the core Retrieval Augmented Generation pipeline:
1. Chunking - split resume into meaningful segments
2. Embedding - convert chunks to vectors
3. Storing - save vectors in ChromaDB
4. Retrieval - find most relevant chunks for a query
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def build_vectorstore(text):
    """
    Build a ChromaDB vector store from text using RAG pipeline.
    
    Process:
    1. Split text into 300-word chunks with 50-word overlap
    2. Generate embeddings for each chunk using MiniLM-L6-v2
    3. Store embeddings in ChromaDB for fast similarity search
    
    Args:
        text: Raw text content from resume
        
    Returns:
        Chroma: Vector store containing embedded text chunks
    """
    # Step 1: Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", ", ", " "]
    )
    chunks = splitter.split_text(text)

    # Step 2 & 3: Embedding + Storing
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
    )

    return vectorstore


def retrieve_relevant_context(vectorstore, query, k=6):
    """
    Retrieve the most relevant resume sections for a given query.
    
    Uses cosine similarity to find the top-K chunks most similar
    to the JD requirements. This is the "Retrieval" in RAG.
    
    Args:
        vectorstore: ChromaDB vector store containing resume chunks
        query: Query string (typically JD requirements combined)
        k: Number of chunks to retrieve (default 6)
        
    Returns:
        str: Concatenated relevant text chunks
    """
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])
