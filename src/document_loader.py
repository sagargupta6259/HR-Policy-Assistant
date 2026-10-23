"""Document loader module for extracting text and metadata from PDF files."""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import pypdf
from langchain_core.documents import Document


def load_pdf_pages(pdf_path: str | Path) -> Tuple[List[Document], Dict[str, Any]]:
    """Load a PDF file and extract text page by page with metadata.
    
    Args:
        pdf_path: Path to the target PDF file.
        
    Returns:
        A tuple of:
            - List of LangChain Document objects (one per non-empty page)
            - Stats dictionary with total_pages, extracted_pages, empty_pages, and errors.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    stats = {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "total_pages": 0,
        "extracted_pages": 0,
        "empty_pages": [],
        "errors": []
    }

    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as e:
        stats["errors"].append(f"Failed to open PDF reader: {str(e)}")
        raise RuntimeError(f"Error reading PDF {path}: {str(e)}") from e

    stats["total_pages"] = len(reader.pages)
    documents: List[Document] = []

    for index, page in enumerate(reader.pages):
        page_num = index + 1
        try:
            text = page.extract_text() or ""
            cleaned_text = text.strip()
            
            if cleaned_text:
                doc = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": path.name,
                        "file_path": str(path.resolve()),
                        "page": page_num,
                        "total_pages": stats["total_pages"]
                    }
                )
                documents.append(doc)
                stats["extracted_pages"] += 1
            else:
                stats["empty_pages"].append(page_num)
        except Exception as e:
            stats["errors"].append(f"Page {page_num} extraction error: {str(e)}")

    return documents, stats
