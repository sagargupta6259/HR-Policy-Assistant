class OrientHistoryBot:
    """Orient BlackSwan History Bot base class."""

    def __init__(self, document_context: str = ""):
        self.document_context = document_context

    def answer_query(self, query: str) -> str:
        """Process user history query and return response."""
        if not self.document_context:
            return "Please provide the history textbook / assignment document first."
        return f"Query received: '{query}'. Ready for RAG / LLM integration."
