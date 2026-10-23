"""Text chunking module for breaking documents into overlapping segments."""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> Tuple[List[Document], Dict[str, Any]]:
    """Split page documents into overlapping chunks with preserved metadata.
    
    Args:
        documents: List of page-level Document objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        
    Returns:
        Tuple of:
            - List of chunked Document objects with unique chunk_id metadata
            - Statistics dictionary
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "•", "Ԃ", "▶", " ", ""]
    )

    chunked_docs: List[Document] = []
    
    for doc in documents:
        page_num = doc.metadata.get("page", 0)
        source = doc.metadata.get("source", "unknown")
        
        splits = splitter.split_text(doc.page_content)
        for chunk_idx, split_text in enumerate(splits):
            chunk_metadata = dict(doc.metadata)
            chunk_metadata.update({
                "chunk_index": chunk_idx,
                "total_chunks_in_page": len(splits),
                "chunk_id": f"{source}_p{page_num}_c{chunk_idx}"
            })
            chunked_docs.append(Document(
                page_content=split_text,
                metadata=chunk_metadata
            ))

    stats = {
        "input_documents": len(documents),
        "total_chunks": len(chunked_docs),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "avg_chunks_per_page": round(len(chunked_docs) / len(documents), 2) if documents else 0
    }

    return chunked_docs, stats
