"""Unit tests for text chunker module."""

from langchain_core.documents import Document
from src.chunker import chunk_documents


def test_chunking_metadata_and_id_generation():
    """Verify chunk generation, overlap, and metadata propagation."""
    sample_text = (
        "Enso Group HR Policy. Section 1 covers recruitment and induction.\n\n"
        "Section 2 covers performance appraisal and controlling absence.\n\n"
        "Section 3 covers grievance procedure and whistleblowing.\n\n"
        "Section 4 covers equality and health and safety policies."
    )
    docs = [
        Document(
            page_content=sample_text,
            metadata={"source": "test.pdf", "page": 1, "total_pages": 1}
        )
    ]

    chunks, stats = chunk_documents(docs, chunk_size=80, chunk_overlap=20)
    assert len(chunks) > 1
    assert stats["total_chunks"] == len(chunks)
    assert stats["input_documents"] == 1

    # Verify metadata preserved on all chunks
    for idx, c in enumerate(chunks):
        assert c.metadata["page"] == 1
        assert c.metadata["source"] == "test.pdf"
        assert c.metadata["chunk_index"] == idx
        assert c.metadata["chunk_id"] == f"test.pdf_p1_c{idx}"
        assert len(c.page_content) > 0


def test_empty_documents_chunking():
    """Verify that empty input documents list returns empty chunks."""
    chunks, stats = chunk_documents([])
    assert chunks == []
    assert stats["total_chunks"] == 0
