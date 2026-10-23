"""CLI ingestion script for parsing the HR Manual PDF and populating ChromaDB."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import PDF_PATH, CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
from src.document_loader import load_pdf_pages
from src.chunker import chunk_documents
from src.vectorstore import index_documents, query_vectorstore


def run_ingestion(pdf_path: Path = PDF_PATH, force: bool = False):
    print("=" * 65)
    print("  Enso Group HR Policy Manual — Ingestion Pipeline")
    print("=" * 65)

    if not pdf_path.exists():
        print(f"\n[ERROR] PDF file not found at: {pdf_path}")
        print("Please place 'hr_manual.pdf' into 'data/raw/' directory.\n")
        sys.exit(1)

    print(f"\n[1/4] Loading PDF: {pdf_path} ...")
    pages, load_stats = load_pdf_pages(pdf_path)
    print(f"      - Total pages found: {load_stats['total_pages']}")
    print(f"      - Pages extracted:   {load_stats['extracted_pages']}")
    print(f"      - Empty pages:       {len(load_stats['empty_pages'])}")
    if load_stats["errors"]:
        print(f"      - Extraction warnings/errors: {len(load_stats['errors'])}")
        for err in load_stats["errors"]:
            print(f"        * {err}")
    else:
        print("      - Status: All pages extracted cleanly (no OCR required).")

    print(f"\n[2/4] Chunking document text ...")
    chunks, chunk_stats = chunk_documents(pages)
    print(f"      - Total chunks produced: {chunk_stats['total_chunks']}")
    print(f"      - Chunk size:            {chunk_stats['chunk_size']} chars")
    print(f"      - Chunk overlap:         {chunk_stats['chunk_overlap']} chars")
    print(f"      - Avg chunks per page:   {chunk_stats['avg_chunks_per_page']}")

    print(f"\n[3/4] Indexing into ChromaDB ({CHROMA_COLLECTION_NAME}) ...")
    index_res = index_documents(
        chunks,
        persist_directory=CHROMA_DB_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        force_reindex=force
    )
    print(f"      - Status: {index_res['status']}")
    print(f"      - Message: {index_res['message']}")
    print(f"      - Total items in vector store: {index_res['total_chunks_indexed']}")
    print(f"      - Storage path: {index_res['persist_directory']}")

    print(f"\n[4/4] Verifying retrieval on indexed store ...")
    test_query = "What is the probation period for employees up to Senior Manager?"
    results = query_vectorstore(test_query, top_k=2)
    if results:
        top_doc, score = results[0]
        print(f"      - Test query: '{test_query}'")
        print(f"      - Top hit page: Page {top_doc.metadata.get('page')}")
        print(f"      - Similarity score: {score:.4f}")
        print(f"      - Excerpt: {top_doc.page_content[:180]}...")
        print("\n[SUCCESS] Ingestion & Vector Index verified successfully!")
    else:
        print("\n[WARNING] No results returned for test query. Please check index.")

    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest HR Manual PDF into ChromaDB.")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH, help="Path to HR manual PDF")
    parser.add_argument("--force", action="store_true", help="Force re-indexing even if already present")
    args = parser.parse_args()

    run_ingestion(pdf_path=args.pdf, force=args.force)
