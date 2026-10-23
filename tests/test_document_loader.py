"""Unit tests for PDF document loader."""

from pathlib import Path
import pytest
from src.config import PDF_PATH
from src.document_loader import load_pdf_pages


def test_missing_pdf_raises_file_not_found():
    """Verify that a non-existent PDF file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_pdf_pages(Path("/non/existent/path/manual.pdf"))


def test_load_real_pdf_extraction():
    """Verify that loading the real 79-page HR manual extracts all pages cleanly."""
    if not PDF_PATH.exists():
        pytest.skip("Source PDF not found in data/raw/")

    docs, stats = load_pdf_pages(PDF_PATH)
    assert stats["total_pages"] == 79
    assert stats["extracted_pages"] == 79
    assert len(docs) == 79
    assert len(stats["empty_pages"]) == 0
    assert len(stats["errors"]) == 0

    # Verify first page metadata
    first_doc = docs[0]
    assert first_doc.metadata["page"] == 1
    assert first_doc.metadata["total_pages"] == 79
    assert first_doc.metadata["source"] == "hr_manual.pdf"
    assert "HR" in first_doc.page_content or "MANUAL" in first_doc.page_content


def test_page_number_consistency():
    """Verify that page metadata accurately tracks physical page indices."""
    if not PDF_PATH.exists():
        pytest.skip("Source PDF not found in data/raw/")

    docs, _ = load_pdf_pages(PDF_PATH)
    page_numbers = [doc.metadata["page"] for doc in docs]
    assert page_numbers == list(range(1, 80))
