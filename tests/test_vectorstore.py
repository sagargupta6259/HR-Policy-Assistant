"""Tests for ChromaDB vector store indexing and duplicate prevention."""

from pathlib import Path
from langchain_core.documents import Document
from src.config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
from src.vectorstore import index_documents, query_vectorstore, get_persistent_client


def test_duplicate_prevention_on_reindexing():
    """Verify that re-indexing an existing collection skips duplication unless forced."""
    test_docs = [
        Document(
            page_content="The company working hours are 9:30 AM to 5:30 PM Monday to Friday.",
            metadata={"source": "test.pdf", "page": 54, "chunk_id": "test_p54_c0"}
        )
    ]
    test_col = "test_duplicate_check_col"
    
    # 1. First index with force=True
    res1 = index_documents(test_docs, persist_directory=CHROMA_DB_DIR, collection_name=test_col, force_reindex=True)
    assert res1["status"] == "success"
    assert res1["total_chunks_indexed"] == 1

    # 2. Second index without force=True should skip
    res2 = index_documents(test_docs, persist_directory=CHROMA_DB_DIR, collection_name=test_col, force_reindex=False)
    assert res2["status"] == "skipped"
    assert "already contains 1 chunks" in res2["message"]

    # Clean up test collection
    client = get_persistent_client(CHROMA_DB_DIR)
    client.delete_collection(name=test_col)


def test_main_collection_retrieval():
    """Verify retrieval against the persistent index."""
    results = query_vectorstore("maternity leave", top_k=2)
    assert len(results) > 0
    top_doc, score = results[0]
    assert 0.0 <= score <= 1.0
    assert "maternity" in top_doc.page_content.lower() or "leave" in top_doc.page_content.lower()
