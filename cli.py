"""Terminal Question-Answer CLI for HR Policy Assistant."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import PDF_PATH, CHROMA_COLLECTION_NAME
from src.rag_pipeline import HRPolicyRAGPipeline


def print_banner():
    print("=" * 70)
    print("       📋 ENSo GROUP HR POLICY ASSISTANT — TERMINAL CLI")
    print("=" * 70)
    print("• Document: Enso Group HR Policy Manual (79 Pages)")
    print("• Vector Store: ChromaDB Collection ('{}')".format(CHROMA_COLLECTION_NAME))
    print("• Type your question below and press Enter.")
    print("• Type 'exit' or 'quit' to exit.")
    print("=" * 70)


def main():
    print_banner()

    pipeline = HRPolicyRAGPipeline()
    chunk_count = pipeline.get_collection_count()

    if chunk_count == 0:
        print("\n[!] Warning: ChromaDB index has 0 chunks.")
        print("    Run 'python ingest.py --force' first to index the HR manual.\n")
    else:
        print(f"[✓] Knowledge Base Active: {chunk_count} chunks indexed.\n")

    while True:
        try:
            user_input = input("HR Question >> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting HR Policy Assistant. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Exiting HR Policy Assistant. Goodbye!")
            break

        print("\nSearching HR Manual and generating policy answer...")
        res = pipeline.answer_question(user_input)

        answer = res.get("answer", "")
        pages = res.get("source_pages", [])
        mode = res.get("mode", "")

        print("\n" + "-" * 70)
        if answer == "Not found":
            print("ANSWER: Not found")
            print("DETAILS: The manual does not contain sufficient evidence to answer this.")
        else:
            print("ANSWER:")
            print(f"  {answer}")
            if pages:
                pages_str = ", ".join(f"Page {p}" for p in pages)
                print(f"\nSOURCE CITATION: {pages_str} (Enso Group HR Policy Manual)")
        
        print(f"ENGINE MODE: {mode}")
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
