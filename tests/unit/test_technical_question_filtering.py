"""
Unit tests for technical question filtering.

This module tests that technical questions are properly filtered out
and not processed as mental health queries.
"""

import pytest
from unittest.mock import AsyncMock
from backend.app.services.query_validator_langgraph import LangGraphQueryValidator
from backend.app.routers.chat_router import handle_random_question
from tests import TestFixtures


class TestTechnicalQuestionFiltering:
    """Tests for filtering technical questions."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider that returns RANDOM_QUESTION for technical queries."""
        mock_provider = AsyncMock()

        async def mock_generate_response(messages):
            message_content = str(messages).lower()

            # Technical questions should be classified as RANDOM_QUESTION
            if any(keyword in message_content for keyword in ['python', 'programming', 'coding', 'technical']):
                return TestFixtures.mock_llm_response(
                    '{"query_type": "RANDOM_QUESTION", "confidence": 0.9, "reasoning": "Technical question detected", "keywords_found": ["python"], "is_crisis": false}'
                )
            else:
                return TestFixtures.mock_llm_response(
                    '{"query_type": "MENTAL_SUPPORT", "confidence": 0.8, "reasoning": "Mental health query", "keywords_found": ["anxious"], "is_crisis": false}'
                )

        mock_provider.generate_response = mock_generate_response
        return mock_provider

    @pytest.fixture
    def validator(self, mock_llm_provider):
        """Create a LangGraph validator with technical question handling."""
        return LangGraphQueryValidator(llm_provider=mock_llm_provider)

    @pytest.mark.asyncio
    async def test_python_question_classification(self, validator):
        """Test that 'what is python' is classified as RANDOM_QUESTION."""
        result = await validator.validate_query("what is python")

        assert result["query_type"] == "random_question"
        assert result["confidence"] >= 0.8
        assert "python" in result["keywords_found"]
        assert result["is_crisis"] == False

    @pytest.mark.asyncio
    async def test_programming_question_classification(self, validator):
        """Test that programming questions are classified as RANDOM_QUESTION."""
        programming_questions = [
            "how do I install python",
            "what is javascript",
            "how to code in java",
            "python programming help",
            "coding tutorial",
            "programming language",
            "software development"
        ]

        for question in programming_questions:
            result = await validator.validate_query(question)

            assert result["query_type"] == "random_question", f"Failed for: {question}"
            assert result["confidence"] >= 0.8, f"Low confidence for: {question}"
            assert result["is_crisis"] == False, f"False crisis for: {question}"

    @pytest.mark.asyncio
    async def test_technical_question_classification(self, validator):
        """Test that technical questions are classified as RANDOM_QUESTION."""
        technical_questions = [
            "how to fix this bug",
            "computer not working",
            "software installation",
            "error in my code",
            "technical support",
            "configuration issue",
            "setup problem"
        ]

        for question in technical_questions:
            result = await validator.validate_query(question)

            assert result["query_type"] == "random_question", f"Failed for: {question}"
            assert result["confidence"] >= 0.8, f"Low confidence for: {question}"
            assert result["is_crisis"] == False, f"False crisis for: {question}"

    @pytest.mark.asyncio
    async def test_mental_health_still_works(self, validator):
        """Test that mental health questions are still classified as MENTAL_SUPPORT."""
        mental_health_questions = [
            "I'm feeling anxious",
            "I need help with depression",
            "having panic attacks",
            "feeling stressed",
            "mental health support"
        ]

        for question in mental_health_questions:
            result = await validator.validate_query(question)

            assert result["query_type"] == "mental_support", f"Failed for: {question}"
            assert result["confidence"] >= 0.8, f"Low confidence for: {question}"
            assert result["is_crisis"] == False, f"False crisis for: {question}"

    @pytest.mark.asyncio
    async def test_mixed_context_questions(self, validator):
        """Test questions that mix technical and emotional context."""
        mixed_questions = [
            ("python is frustrating", "mental_support"),  # Emotional context
            ("I'm stressed about coding", "mental_support"),  # Mental health focus
            ("programming makes me anxious", "mental_support"),  # Emotional state
            ("how to code when depressed", "mental_support"),  # Mental health focus
            ("python syntax help", "random_question"),  # Pure technical
            ("install python", "random_question"),  # Pure technical
        ]

        for question, expected_type in mixed_questions:
            result = await validator.validate_query(question)

            assert result["query_type"] == expected_type, f"Failed for: {question}"
            assert result["confidence"] >= 0.7, f"Low confidence for: {question}"

    @pytest.mark.asyncio
    async def test_handle_random_question_technical(self):
        """Test handling of technical random questions."""
        validation_result = {
            "query_type": "random_question",
            "confidence": 0.9,
            "suggestions": ["Provide technical guidance"]
        }

        # Test technical question handling
        response = await handle_random_question("how do I install python", validation_result)

        assert "technical" in response.lower()
        assert "programming" in response.lower()
        assert "install python" in response
        assert "mental health" in response.lower()

    @pytest.mark.asyncio
    async def test_handle_random_question_general(self):
        """Test handling of general random questions."""
        validation_result = {
            "query_type": "random_question",
            "confidence": 0.8,
            "suggestions": ["Provide general information"]
        }

        # Test general question handling
        response = await handle_random_question("what's the weather", validation_result)

        assert "question" in response.lower()
        assert "mental health" in response.lower()
        assert "weather" in response
        assert "technical" not in response.lower()

    @pytest.mark.asyncio
    async def test_chat_router_integration(self, mock_llm_provider):
        """Test chat router integration with technical question filtering."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from backend.app.routers.chat_router import router, get_llm_service, get_query_validator

        app = FastAPI()

        # Mock services
        mock_llm_service = AsyncMock()
        mock_llm_service.generate_response = AsyncMock(return_value="Mental health response")

        mock_validator = AsyncMock()
        mock_validator.validate_query = AsyncMock(return_value={
            "query_type": "random_question",
            "confidence": 0.9,
            "reasoning": "Technical question detected",
            "is_crisis": False,
            "routing_decision": "general_assistance"
        })

        app.dependency_overrides[get_llm_service] = lambda: mock_llm_service
        app.dependency_overrides[get_query_validator] = lambda: mock_validator
        app.include_router(router)

        client = TestClient(app)

        # Test technical question
        response = client.post(
            "/api/chat",
            json={"message": "what is python"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should contain technical question handling response
        assert "technical" in data["response"].lower()
        assert "python" in data["response"]
        assert "mental health" in data["response"].lower()

    @pytest.mark.asyncio
    async def test_edge_cases(self, validator):
        """Test edge cases for technical question classification."""
        edge_cases = [
            ("python", "random_question"),  # Single word
            ("python programming", "random_question"),  # Two words
            ("i love python", "random_question"),  # Positive context
            ("python is hard", "random_question"),  # Difficulty context
            ("python makes me happy", "mental_support"),  # Emotional context
            ("stressed about python", "mental_support"),  # Mental health context
        ]

        for question, expected_type in edge_cases:
            result = await validator.validate_query(question)

            assert result["query_type"] == expected_type, f"Failed for: {question}"
            assert result["confidence"] >= 0.7, f"Low confidence for: {question}"

    @pytest.mark.asyncio
    async def test_confidence_scoring(self, validator):
        """Test confidence scoring for technical questions."""
        test_cases = [
            ("python", 0.9),  # High confidence - clear technical term
            ("programming", 0.9),  # High confidence - clear technical term
            ("how to code", 0.85),  # Medium-high confidence - technical pattern
            ("computer", 0.8),  # Medium confidence - could be technical or general
            ("software", 0.8),  # Medium confidence - technical context
        ]

        for question, expected_confidence in test_cases:
            result = await validator.validate_query(question)

            assert result["query_type"] == "random_question", f"Wrong type for: {question}"
            assert result["confidence"] >= expected_confidence, f"Low confidence for: {question}"
            assert result["confidence"] <= 1.0, f"Confidence too high for: {question}"