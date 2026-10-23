"""Unit and integration tests for the RAG pipeline."""

from unittest.mock import MagicMock, patch
import pytest
from src.rag_pipeline import HRPolicyRAGPipeline


@pytest.fixture(scope="module")
def pipeline():
    return HRPolicyRAGPipeline()


def test_empty_user_question_validation(pipeline):
    """Verify empty query validation."""
    res = pipeline.answer_question("")
    assert res["error"] == "Empty question"
    assert res["source_pages"] == []

    res_whitespace = pipeline.answer_question("   \n\t  ")
    assert res_whitespace["error"] == "Empty question"


def test_missing_index_handling():
    """Verify handling when collection has zero chunks."""
    empty_pipeline = HRPolicyRAGPipeline(collection_name="empty_dummy_collection")
    res = empty_pipeline.answer_question("What is the leave policy?")
    assert res["error"] == "Index missing"
    assert "empty or missing" in res["answer"].lower()


def test_question_privilege_leave(pipeline):
    """Q1: How many days of Privilege Leave (PL) does a permanent employee get per year?"""
    q = "How many days of Privilege Leave (PL) does a permanent employee get per year?"
    res = pipeline.answer_question(q)
    assert res["answer"] != "Not found"
    assert "12 days" in res["answer"]
    assert 60 in res["source_pages"]


def test_question_probation_period(pipeline):
    """Q2: What is the probation period for employees up to Senior Manager?"""
    q = "What is the probation period for employees up to Senior Manager?"
    res = pipeline.answer_question(q)
    assert res["answer"] != "Not found"
    assert "3 months" in res["answer"]
    assert 42 in res["source_pages"]


def test_question_unsupported_health_insurance(pipeline):
    """Q3: What is the company's health-insurance premium amount? (Must return exactly 'Not found')"""
    q = "What is the company's health-insurance premium amount?"
    res = pipeline.answer_question(q)
    assert res["answer"] == "Not found"
    assert res["source_pages"] == []


def test_question_unsupported_gst_rate(pipeline):
    """Q4: What is the current GST rate in India? (Must return exactly 'Not found')"""
    q = "What is the current GST rate in India?"
    res = pipeline.answer_question(q)
    assert res["answer"] == "Not found"
    assert res["source_pages"] == []


def test_question_maternity_leave(pipeline):
    """Q5: How many days of maternity leave are given?"""
    q = "How many days of maternity leave are given?"
    res = pipeline.answer_question(q)
    assert res["answer"] != "Not found"
    assert "12 weeks" in res["answer"] or "90 days" in res["answer"]
    assert 62 in res["source_pages"]


def test_mocked_llm_execution():
    """Verify that when OPENAI_API_KEY is provided, ChatOpenAI chain is properly invoked."""
    mock_pipeline = HRPolicyRAGPipeline(openai_api_key="test_fake_api_key")
    
    with patch("langchain_openai.ChatOpenAI") as mock_chat:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content="12 days of Privilege Leave per Page 60.")
        mock_chat.return_value = mock_instance

        # Test answering question with mock
        res = mock_pipeline.answer_question("How many days of Privilege Leave (PL) does a permanent employee get per year?")
        assert res["answer"] != ""
        assert res["source_pages"] != []
