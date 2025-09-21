#!/usr/bin/env python3
"""
Unit tests for the LangGraph Query Validator Service

This module contains unit tests for the LangGraph-based query validator.
Moved from backend/test_langgraph_validator.py to tests/unit/ for better organization.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from backend.app.services.query_validator_langgraph import LangGraphQueryValidator
from backend.app.services.langgraph_state import QueryType, CrisisSeverity


class TestLangGraphQueryValidatorUnit:
    """Unit tests for LangGraphQueryValidator."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        mock_provider = AsyncMock()

        async def mock_generate_response(messages):
            return '{"query_type": "MENTAL_SUPPORT", "confidence": 0.8, "reasoning": "Mock response", "keywords_found": ["test"], "is_crisis": false}'

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

    def test_make_routing_decision(self, validator):
        """Test routing decision logic."""
        # Test crisis routing
        crisis_state = {
            "crisis_assessment": {"is_crisis": True}
        }
        decision = validator._make_routing_decision(crisis_state)
        assert decision == "crisis_intervention"

        # Test mental support routing
        mental_support_state = {
            "classification": {"query_type": QueryType.MENTAL_SUPPORT}
        }
        decision = validator._make_routing_decision(mental_support_state)
        assert decision == "mental_health_support"

        # Test random question routing
        random_question_state = {
            "classification": {"query_type": QueryType.RANDOM_QUESTION}
        }
        decision = validator._make_routing_decision(random_question_state)
        assert decision == "general_assistance"

    def test_generate_final_response(self, validator):
        """Test final response generation."""
        # Test crisis response
        crisis_state = {
            "crisis_assessment": {"is_crisis": True}
        }
        response = validator._generate_final_response(crisis_state)
        assert "crisis situation" in response
        assert "112" in response

        # Test mental support response
        mental_support_state = {
            "classification": {"query_type": QueryType.MENTAL_SUPPORT}
        }
        response = validator._generate_final_response(mental_support_state)
        assert "mental health support" in response

        # Test random question response
        random_question_state = {
            "classification": {"query_type": QueryType.RANDOM_QUESTION}
        }
        response = validator._generate_final_response(random_question_state)
        assert "question" in response

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self):
        """Test error handling in workflow."""
        # Test with None LLM provider
        validator = LangGraphQueryValidator(llm_provider=None)

        result = await validator.validate_query("Test query")

        assert result["query_type"] == "unclear"
        assert len(result["errors"]) > 0
        assert result["requires_human_intervention"]

    @pytest.mark.asyncio
    async def test_workflow_with_different_query_types(self, validator):
        """Test workflow with different query types."""
        test_cases = [
            ("I'm feeling really anxious today", "mental_support"),
            ("What's the weather like today?", "random_question"),
            ("I need help with depression", "mental_support"),
            ("", "unclear"),
            ("xyz", "unclear")
        ]

        for query, expected_type in test_cases:
            result = await validator.validate_query(query)
            assert result["query_type"] == expected_type

    def test_validator_initialization(self):
        """Test validator initialization."""
        # Test with LLM provider
        mock_provider = Mock()
        validator = LangGraphQueryValidator(llm_provider=mock_provider)
        assert validator.llm_provider == mock_provider
        assert validator.is_initialized

        # Test without LLM provider
        validator = LangGraphQueryValidator(llm_provider=None)
        assert validator.llm_provider is None
        assert not validator.is_initialized

    @pytest.mark.asyncio
    async def test_workflow_state_transitions(self, mock_llm_provider):
        """Test state transitions throughout the workflow."""
        validator = LangGraphQueryValidator(llm_provider=mock_llm_provider)

        # Create initial state
        initial_state = {
            "query": "Test query",
            "user_id": "test_user",
            "conversation_history": []
        }

        # Test routing decision
        routing_decision = validator._make_routing_decision(initial_state)
        assert "routing_decision" in initial_state or routing_decision == "standard_processing"

        # Test final response generation
        final_response = validator._generate_final_response(initial_state)
        assert isinstance(final_response, str)
        assert len(final_response) > 0


class TestLangGraphQueryValidatorIntegration:
    """Integration tests for LangGraphQueryValidator."""

    @pytest.mark.asyncio
    async def test_end_to_end_validation(self, mock_llm_provider):
        """Test end-to-end query validation process."""
        validator = LangGraphQueryValidator(llm_provider=mock_llm_provider)

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
    async def test_workflow_with_conversation_history(self, mock_llm_provider):
        """Test workflow with conversation history."""
        validator = LangGraphQueryValidator(llm_provider=mock_llm_provider)

        query = "I'm still feeling anxious"
        conversation_history = [
            {"role": "user", "text": "I'm feeling anxious", "timestamp": datetime.now()},
            {"role": "assistant", "text": "I understand you're feeling anxious. Can you tell me more?", "timestamp": datetime.now()},
            {"role": "user", "text": "Yes, it's been getting worse", "timestamp": datetime.now()}
        ]

        result = await validator.validate_query(query, conversation_history=conversation_history)

        assert isinstance(result, dict)
        assert "query_type" in result
        assert result["query_type"] == "mental_support"

    @pytest.mark.asyncio
    async def test_workflow_performance(self, validator):
        """Test workflow performance."""
        import time

        start_time = time.time()
        result = await validator.validate_query("Test query")
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within reasonable time (mocked, so very fast)
        assert execution_time < 1.0  # Less than 1 second
        assert isinstance(result, dict)


class TestLangGraphWorkflow:
    """Test the new LangGraph workflow with conversation filtering."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider for testing."""
        mock_provider = AsyncMock()
        mock_provider.generate_response = AsyncMock()

        # Mock classification response for mental support query
        mock_provider.generate_response.return_value = '''
        {
            "query_type": "mental_support",
            "confidence": 0.85,
            "reasoning": "Query contains emotional indicators",
            "keywords_found": ["feeling", "anxious"],
            "is_crisis": false
        }
        '''
        return mock_provider

    @pytest.fixture
    def mock_llm_provider_random(self):
        """Create a mock LLM provider for random questions."""
        mock_provider = AsyncMock()
        mock_provider.generate_response = AsyncMock()

        # Mock classification response for random question
        mock_provider.generate_response.return_value = '''
        {
            "query_type": "random_question",
            "confidence": 0.92,
            "reasoning": "Query is about programming",
            "keywords_found": ["python", "install"],
            "is_crisis": false
        }
        '''
        return mock_provider

    @pytest.fixture
    def mock_llm_provider_crisis(self):
        """Create a mock LLM provider for crisis situations."""
        mock_provider = AsyncMock()
        mock_provider.generate_response = AsyncMock()

        # Mock classification response for crisis
        mock_provider.generate_response.return_value = '''
        {
            "query_type": "crisis",
            "confidence": 0.95,
            "reasoning": "Query indicates immediate danger",
            "keywords_found": ["suicide", "end it"],
            "is_crisis": true
        }
        '''
        return mock_provider

    @pytest.mark.asyncio
    async def test_workflow_mental_support_proceeds(self, mock_llm_provider):
        """Test that mental support queries proceed to conversation."""
        validator = LangGraphQueryValidator(llm_provider=mock_llm_provider)

        result = await validator.execute_workflow("I'm feeling really anxious and stressed")

        assert result["should_proceed_to_conversation"] is True
        assert result["query_type"] == "mental_support"
        assert result["routing_decision"] == "mental_health_support"

    @pytest.mark.asyncio
    async def test_workflow_random_question_filtered(self, mock_llm_provider_random):
        """Test that random questions are filtered out."""
        validator = LangGraphQueryValidator(llm_provider=mock_llm_provider_random)

        result = await validator.execute_workflow("How do I install Python on my computer?")

        assert result["should_proceed_to_conversation"] is False
        assert result["query_type"] == "random_question"
        assert result["routing_decision"] == "random_question_filtered"

    @pytest.mark.asyncio
    async def test_workflow_crisis_always_proceeds(self, mock_llm_provider_crisis):
        """Test that crisis situations always proceed to conversation."""
        validator = LangGraphQueryValidator(llm_provider=mock_llm_provider_crisis)

        result = await validator.execute_workflow("I want to end it all, I can't take this anymore")

        assert result["should_proceed_to_conversation"] is True
        assert result["query_type"] == "crisis"
        assert result["routing_decision"] == "crisis_intervention"
        assert result["is_crisis"] is True

    @pytest.mark.asyncio
    async def test_workflow_unclear_filtered(self):
        """Test that unclear queries are filtered out."""
        validator = LangGraphQueryValidator(llm_provider=None)  # No LLM provider

        result = await validator.execute_workflow("xyz maybe")

        assert result["should_proceed_to_conversation"] is False
        assert result["query_type"] == "unclear"
        assert result["routing_decision"] == "fallback_processing"

    @pytest.mark.asyncio
    async def test_workflow_fallback_behavior(self):
        """Test fallback behavior when workflow fails."""
        validator = LangGraphQueryValidator(llm_provider=None)

        result = await validator.execute_workflow("")

        assert result["should_proceed_to_conversation"] is False
        assert result["query_type"] == "unclear"
        assert result["routing_decision"] == "fallback_processing"
        assert "errors" in result
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_determine_conversation_proceeding_logic(self, mock_llm_provider):
        """Test the conversation proceeding logic directly."""
        validator = LangGraphQueryValidator(llm_provider=mock_llm_provider)

        # Test mental support - should proceed
        mental_support_state = {
            "classification": {"query_type": QueryType.MENTAL_SUPPORT},
            "crisis_assessment": {"is_crisis": False}
        }
        should_proceed = validator._determine_conversation_proceeding(mental_support_state)
        assert should_proceed is True

        # Test random question - should not proceed
        random_question_state = {
            "classification": {"query_type": QueryType.RANDOM_QUESTION},
            "crisis_assessment": {"is_crisis": False}
        }
        should_proceed = validator._determine_conversation_proceeding(random_question_state)
        assert should_proceed is False

        # Test crisis - should always proceed
        crisis_state = {
            "classification": {"query_type": QueryType.RANDOM_QUESTION},
            "crisis_assessment": {"is_crisis": True}
        }
        should_proceed = validator._determine_conversation_proceeding(crisis_state)
        assert should_proceed is True

    def test_new_routing_decision_logic(self):
        """Test the updated routing decision logic."""
        validator = LangGraphQueryValidator(llm_provider=None)

        # Test random question routing
        random_question_state = {
            "classification": {"query_type": QueryType.RANDOM_QUESTION}
        }
        decision = validator._make_routing_decision(random_question_state)
        assert decision == "random_question_filtered"

        # Test mental support routing (unchanged)
        mental_support_state = {
            "classification": {"query_type": QueryType.MENTAL_SUPPORT}
        }
        decision = validator._make_routing_decision(mental_support_state)
        assert decision == "mental_health_support"