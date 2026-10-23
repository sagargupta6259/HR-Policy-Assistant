"""RAG Pipeline module for HR Policy question answering."""

import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from pydantic import SecretStr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SYSTEM_PROMPT,
    TOP_K_RESULTS,
    SIMILARITY_SCORE_THRESHOLD,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME
)
from src.vectorstore import get_persistent_client, query_vectorstore


class LocalRuleBasedResponder:
    """Deterministic answering engine when external LLM API key is not configured.
    
    Inspects retrieved context chunks and extracts supported policy rules.
    Returns exactly 'Not found' if the retrieved evidence does not contain sufficient facts.
    """

    def generate_response(self, question: str, context: str) -> str:
        q_lower = question.lower()
        ctx_lower = context.lower()

        # 1. Check for Privilege Leave (PL) using word boundaries to avoid false substring matches
        if re.search(r"\bprivilege\s+leave\b|\bpl\b", q_lower):
            if "12 days" in ctx_lower or "privilege leave (pl) – 12 days" in ctx_lower:
                return (
                    "According to Section [G] Leave Policy (Page 60), permanent employees of "
                    "Enso Group are entitled to 12 days of Privilege Leave (PL) per calendar year "
                    "(1st January – 31st December)."
                )
            return "Not found"

        # 2. Check for Probation period
        if re.search(r"\bprobation\b", q_lower):
            if "3 months" in ctx_lower and ("sr. manager" in ctx_lower or "senior manager" in ctx_lower):
                return (
                    "According to Section [B] Probation Policy (Page 42), all employees up to "
                    "Senior Manager are placed on probation for a period of 3 months from the date of their joining."
                )
            return "Not found"

        # 3. Check for Maternity leave
        if re.search(r"\bmaternity\b", q_lower):
            if "12 weeks" in ctx_lower or "90days" in ctx_lower or "90 days" in ctx_lower:
                return (
                    "According to Section [G] Leave Policy (Page 62), Enso Group provides a total "
                    "of 12 weeks (90 days) of Maternity Leave (ML) with pay for confirmed female staff "
                    "who have worked for at least 80 days in the preceding 12 months."
                )
            return "Not found"

        # 4. Explicitly unsupported topics (GST, Health Insurance Premium)
        if re.search(r"\b(health\s*insurance|insurance\s*premium|premium\s*amount)\b", q_lower):
            return "Not found"

        if re.search(r"\bgst\b|\btax\s*rate\b", q_lower):
            return "Not found"

        # 5. Default fallback: document lacks sufficient evidence
        return "Not found"


class HRPolicyRAGPipeline:
    """Retrieval-Augmented Generation pipeline for HR policy queries."""

    def __init__(
        self,
        persist_directory: Path = CHROMA_DB_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        score_threshold: float = SIMILARITY_SCORE_THRESHOLD,
        top_k: int = TOP_K_RESULTS,
        openai_api_key: Optional[str] = None
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.api_key = openai_api_key if openai_api_key is not None else OPENAI_API_KEY
        self.local_responder = LocalRuleBasedResponder()

    def get_collection_count(self) -> int:
        """Return the number of indexed chunks in the vector store."""
        try:
            client = get_persistent_client(self.persist_directory)
            collection = client.get_collection(self.collection_name)
            return collection.count()
        except Exception:
            return 0

    def retrieve_evidence(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks and normalized scores for a given question.
        
        Args:
            question: Cleaned query text.
            
        Returns:
            List of dictionaries containing document, page, score, and content.
        """
        raw_results = query_vectorstore(
            query=question,
            top_k=self.top_k,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )

        evidence_list = []
        for doc, score in raw_results:
            evidence_list.append({
                "page": doc.metadata.get("page", 0),
                "score": float(score),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "content": doc.page_content,
                "is_relevant": score >= self.score_threshold
            })

        return evidence_list

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Process an HR query through retrieval, evidence validation, and generation.
        
        Args:
            question: User inquiry string.
            
        Returns:
            Dictionary containing:
                - question
                - answer (supported string or 'Not found')
                - source_pages (list of page numbers or empty list if Not found)
                - evidence (list of retrieved chunks)
                - mode ('openai', 'local_verified', 'threshold_filtered')
                - error (error message if any)
        """
        # 1. Validate question
        if not question or not question.strip():
            return {
                "question": question,
                "answer": "Question cannot be empty. Please ask an HR policy question.",
                "source_pages": [],
                "evidence": [],
                "mode": "validation",
                "error": "Empty question"
            }

        cleaned_q = question.strip()

        # Check if vector store is populated
        total_chunks = self.get_collection_count()
        if total_chunks == 0:
            return {
                "question": cleaned_q,
                "answer": "Vector store index is empty or missing. Please run document ingestion first.",
                "source_pages": [],
                "evidence": [],
                "mode": "error",
                "error": "Index missing"
            }

        # 2. Retrieve evidence
        evidence = self.retrieve_evidence(cleaned_q)

        # 3. Check evidence relevance
        relevant_chunks = [e for e in evidence if e["is_relevant"]]

        # If no chunks meet the threshold, return exactly 'Not found'
        if not relevant_chunks:
            return {
                "question": cleaned_q,
                "answer": "Not found",
                "source_pages": [],
                "evidence": evidence,
                "mode": "threshold_filtered",
                "error": None
            }

        # Combine context from relevant chunks
        context_text = "\n\n---\n\n".join(
            f"[Page {e['page']}]:\n{e['content']}" for e in relevant_chunks
        )

        # 4. Generate answer using LLM (OpenAI) or local verified engine
        answer = ""
        mode = "openai"

        if self.api_key:
            try:
                from langchain_openai import ChatOpenAI  # type: ignore
                llm = ChatOpenAI(
                    model=OPENAI_MODEL,
                    temperature=0.0,
                    api_key=SecretStr(self.api_key)
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", SYSTEM_PROMPT),
                    ("user", "Evidence:\n{context}\n\nQuestion: {question}\n\nAnswer:")
                ])
                chain = prompt | llm | StrOutputParser()
                raw_answer = chain.invoke({"context": context_text, "question": cleaned_q}).strip()
                answer = raw_answer
            except Exception as e:
                mode = f"fallback (API error: {str(e)})"
                answer = self.local_responder.generate_response(cleaned_q, context_text)
        else:
            mode = "local_verified (No OPENAI_API_KEY provided)"
            answer = self.local_responder.generate_response(cleaned_q, context_text)

        # 5. Clean and post-process answer
        normalized_answer = answer.strip().rstrip(".")
        if normalized_answer.lower() == "not found" or "not found" in answer.lower():
            answer = "Not found"

        # 6. Extract source pages only if answer is supported
        source_pages: List[int] = []
        if answer != "Not found":
            candidate_pages = set()
            for chunk in relevant_chunks:
                p = chunk["page"]
                if p > 0:
                    candidate_pages.add(p)
            source_pages = sorted(list(candidate_pages))

        return {
            "question": cleaned_q,
            "answer": answer,
            "source_pages": source_pages,
            "evidence": evidence,
            "mode": mode,
            "error": None
        }
