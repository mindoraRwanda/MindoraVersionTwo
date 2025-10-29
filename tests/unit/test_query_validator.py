"""
Unit tests for the Query Validator service.

This module contains unit tests for the LangGraph-based query validators.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from backend.app.services.query_validator_langgraph import LangGraphQueryValidator
from backend.app.services.langgraph_state import QueryType
from tests import TestUtils, TestFixtures


class TestLangGraphQueryValidator:
    """Tests for the LangGraph-based QueryValidator."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        mock_provider = AsyncMock()

        async def mock_generate_response(messages):
            return TestFixtures.mock_llm_response("Mock classification response")

        mock_provider.generate_response = mock_generate_response
        return mock_provider

    @pytest.fixture
    def validator(self, mock_llm_provider):
        """Create a LangGraphQueryValidator instance."""
        return LangGraphQueryValidator(llm_provider=mock_llm_provider)

    @pytest.mark.asyncio
    async def test_validate_query_with_llm(self, validator):
        """Test query validation with LLM provider."""
        result = await validator.validate_query("I'm feeling anxious")

        assert "query_type" in result
        assert "confidence" in result
        assert "routing_decision" in result

    @pytest.mark.asyncio
    async def test_validate_query_without_llm(self):
        """Test query validation without LLM provider."""
        validator = LangGraphQueryValidator(llm_provider=None)
        result = await validator.validate_query("Test query")

        assert result["query_type"] == "unclear"
        assert result["confidence"] == 0.5
        assert "fallback" in result["routing_decision"]

    @pytest.mark.asyncio
    async def test_crisis_assessment(self, validator):
        """Test crisis assessment functionality."""
        result = await validator.validate_query("I want to end it all")

        assert "is_crisis" in result
        assert "crisis_severity" in result

    def test_extract_results(self, validator):
        """Test result extraction from workflow state."""
        mock_state = {
            "classification": {
                "query_type": QueryType.MENTAL_SUPPORT,
                "confidence": 0.9,
                "reasoning": "Test reasoning",
                "keywords_found": ["anxious"]
            },
            "crisis_assessment": {
                "is_crisis": False,
                "severity": "low"
            },
            "suggestions": {
                "suggestions": ["Test suggestion"],
                "routing_priority": "medium"
            },
            "routing_decision": "mental_health_support",
            "final_response": "Test response",
            "processing_timestamp": datetime.now(),
            "errors": []
        }

        results = validator._extract_results(mock_state)

        assert results["query_type"] == "mental_support"
        assert results["confidence"] == 0.9
        assert results["routing_decision"] == "mental_health_support"


class TestQueryValidationIntegration:
    """Integration tests for query validation."""

    @pytest.mark.asyncio
    async def test_end_to_end_validation(self, mock_llm_service):
        """Test end-to-end query validation process."""
        validator = LangGraphQueryValidator(llm_provider=mock_llm_service.llm_provider)

        test_queries = [
            "I'm feeling really anxious today",
            "What's the weather like?",
            "I need help with depression"
        ]

        for query in test_queries:
            result = await validator.validate_query(query)

            # Basic validation
            assert isinstance(result, dict)
            assert "query_type" in result
            assert "confidence" in result
            assert "routing_decision" in result

            # Type validation
            assert isinstance(result["confidence"], (int, float))
            assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in query validation."""
        # Test with None LLM provider
        validator = LangGraphQueryValidator(llm_provider=None)

        result = await validator.validate_query("Test query")

        assert result["query_type"] == "unclear"
        assert len(result["errors"]) > 0
        assert result["requires_human_intervention"]

    @pytest.mark.asyncio
    async def test_workflow_state_management(self, mock_llm_provider):
        """Test workflow state management."""
        validator = LangGraphQueryValidator(llm_provider=mock_llm_provider)

        # Create initial state
        initial_state = {
            "query": "Test query",
            "user_id": "test_user",
            "conversation_history": []
        }

        # Test state updates
        classification = {
            "query_type": QueryType.MENTAL_SUPPORT,
            "confidence": 0.9,
            "reasoning": "Test",
            "keywords_found": [],
            "is_crisis": False,
            "crisis_severity": None,
            "requires_human_intervention": False
        }

        updated_state = validator._make_routing_decision(initial_state)
        assert "routing_decision" in updated_state or updated_state == "standard_processing"