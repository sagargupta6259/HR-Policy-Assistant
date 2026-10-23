"""ChromaDB vector store management module."""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import chromadb
import chromadb.utils.embedding_functions as ef
from pydantic import SecretStr
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from src.config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    TOP_K_RESULTS
)


class ChromaDefaultEmbeddings(Embeddings):
    """Wrapper around Chroma's built-in ONNX all-MiniLM-L6-v2 embedding function."""

    def __init__(self):
        self._fn = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raw_embeddings = self._fn(texts)
        return [[float(val) for val in emb] for emb in raw_embeddings]

    def embed_query(self, text: str) -> List[float]:
        raw_embedding = self._fn([text])[0]
        return [float(val) for val in raw_embedding]


def get_embedding_function() -> Embeddings:
    """Return the configured embedding function.
    
    Defaults to Chroma's built-in local ONNX embedding function (no API key required).
    If OPENAI_API_KEY is available and configured, OpenAI embeddings can be used.
    """
    if EMBEDDING_PROVIDER == "openai" and OPENAI_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore
            return OpenAIEmbeddings(api_key=SecretStr(OPENAI_API_KEY))
        except (ImportError, Exception):
            pass
    
    return ChromaDefaultEmbeddings()


def get_persistent_client(persist_directory: Path = CHROMA_DB_DIR) -> Any:
    """Initialize or retrieve a persistent ChromaDB client."""
    persist_directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_directory))


def get_vectorstore(
    persist_directory: Path = CHROMA_DB_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_function: Optional[Embeddings] = None
) -> Chroma:
    """Retrieve or initialize the LangChain Chroma vector store."""
    if embedding_function is None:
        embedding_function = get_embedding_function()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(persist_directory)
    )


def index_documents(
    documents: List[Document],
    persist_directory: Path = CHROMA_DB_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    force_reindex: bool = False
) -> Dict[str, Any]:
    """Index chunks into ChromaDB, preventing duplicates unless forced.
    
    Args:
        documents: List of chunk Document objects.
        persist_directory: Target Chroma directory.
        collection_name: Target collection name.
        force_reindex: If True, clear and recreate collection.
        
    Returns:
        Indexing statistics dictionary.
    """
    client = get_persistent_client(persist_directory)
    
    # If forced, remove existing collection
    if force_reindex:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name)
    existing_count = collection.count()

    # Prevent duplicate ingestion if already indexed and not forced
    if existing_count > 0 and not force_reindex:
        return {
            "status": "skipped",
            "message": f"Collection '{collection_name}' already contains {existing_count} chunks. Skipping re-indexing.",
            "total_chunks_indexed": existing_count,
            "collection_name": collection_name,
            "persist_directory": str(persist_directory)
        }

    # Generate deterministic IDs for chunks
    ids = [
        doc.metadata.get("chunk_id", f"doc_chunk_{i}")
        for i, doc in enumerate(documents)
    ]
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    # Initialize LangChain Chroma store and add texts in batches
    emb_fn = get_embedding_function()
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=emb_fn
    )

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        vectorstore.add_texts(
            texts=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )

    final_count = vectorstore._collection.count()
    return {
        "status": "success",
        "message": f"Successfully indexed {final_count} chunks into '{collection_name}'.",
        "total_chunks_indexed": final_count,
        "collection_name": collection_name,
        "persist_directory": str(persist_directory)
    }


def query_vectorstore(
    query: str,
    top_k: int = TOP_K_RESULTS,
    persist_directory: Path = CHROMA_DB_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME
) -> List[Tuple[Document, float]]:
    """Query the vector store with normalized similarity scores (0.0 to 1.0).
    
    Returns:
        List of (Document, score) tuples.
    """
    vectorstore = get_vectorstore(persist_directory, collection_name)
    raw_results = vectorstore.similarity_search_with_score(query, k=top_k)
    normalized_results = []
    for doc, dist in raw_results:
        # Cosine distance is in [0, 2]; normalized similarity = max(0.0, 1.0 - dist / 2.0)
        norm_score = max(0.0, 1.0 - (float(dist) / 2.0))
        normalized_results.append((doc, norm_score))
    return normalized_results
