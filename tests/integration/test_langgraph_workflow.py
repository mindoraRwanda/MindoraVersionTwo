"""
Integration tests for the LangGraph workflow.

This module contains integration tests for the complete LangGraph workflow
including all components working together.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from backend.app.services.query_validator_langgraph import LangGraphQueryValidator
from backend.app.services.langgraph_state import QueryValidationState, QueryType, CrisisSeverity
from tests import TestUtils, TestFixtures


class TestLangGraphWorkflow:
    """Integration tests for the LangGraph workflow."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider that returns realistic responses."""
        mock_provider = AsyncMock()

        async def mock_generate_response(messages):
            # Extract content from messages to check for keywords
            message_content = ""
            for msg in messages:
                if hasattr(msg, 'content'):
                    message_content += msg.content + " "
                else:
                    message_content += str(msg) + " "

            # Determine the type of request based on the prompt content
            if "Query to classify:" in message_content:
                # This is a classification request
                query = message_content.split("Query to classify:")[-1].strip().strip('"')

                # Check for crisis keywords in the actual query
                crisis_keywords = ["kill myself", "suicide", "end my life", "don't want to live", "harm myself", "suicidal thoughts"]
                if any(keyword in query.lower() for keyword in crisis_keywords):
                    return '{"query_type": "CRISIS", "confidence": 0.95, "reasoning": "Crisis indicators detected", "keywords_found": ["suicide"], "is_crisis": true}'

                # Check for random question keywords in the actual query
                random_keywords = ["weather", "python", "programming", "install", "computer", "software", "what is", "how do", "capital", "population"]
                if any(keyword in query.lower() for keyword in random_keywords):
                    return '{"query_type": "RANDOM_QUESTION", "confidence": 0.9, "reasoning": "Random question detected", "keywords_found": ["weather"], "is_crisis": false}'

                # Check for unclear queries in the actual query
                unclear_keywords = ["xyz", "abc", "maybe", "idk", "not sure", "random"]
                if any(keyword in query.lower() for keyword in unclear_keywords) or query.strip() == "":
                    return '{"query_type": "UNCLEAR", "confidence": 0.8, "reasoning": "Unclear query", "keywords_found": [], "is_crisis": false}'

                # Default to mental support for emotional content in the actual query
                mental_keywords = ["feeling", "anxious", "sad", "angry", "stressed", "depressed", "worried", "overwhelmed", "help"]
                if any(keyword in query.lower() for keyword in mental_keywords):
                    return '{"query_type": "MENTAL_SUPPORT", "confidence": 0.9, "reasoning": "Mental health indicators detected", "keywords_found": ["anxious"], "is_crisis": false}'

                # Default fallback
                return '{"query_type": "MENTAL_SUPPORT", "confidence": 0.7, "reasoning": "Default classification", "keywords_found": [], "is_crisis": false}'

            elif "Query to assess for crisis:" in message_content:
                # This is a crisis assessment request
                query = message_content.split("Query to assess for crisis:")[-1].strip().strip('"')
                crisis_keywords = ["kill myself", "suicide", "end my life", "don't want to live", "harm myself"]
                if any(keyword in query.lower() for keyword in crisis_keywords):
                    return '{"is_crisis": true, "confidence": 0.95, "crisis_indicators": ["suicidal ideation"], "severity": "critical", "recommended_action": "immediate_intervention"}'
                else:
                    return '{"is_crisis": false, "confidence": 0.8, "crisis_indicators": [], "severity": "low", "recommended_action": "monitor"}'

            elif "Analyze the emotions in this query:" in message_content:
                # This is an emotion detection request
                return '{"detected_emotion": "anxiety", "confidence": 0.8, "emotion_scores": {"anxiety": 0.9, "fear": 0.6}, "reasoning": "Emotional content detected", "intensity": "high", "context_relevance": "high"}'

            elif "Generate suggestions for this query:" in message_content:
                # This is a suggestions request
                return '{"suggestions": ["Listen actively", "Validate feelings"], "routing_priority": "high", "requires_human_intervention": false, "follow_up_questions": ["How long have you felt this way?"], "next_best_action": "mental_health_support"}'

            elif "Analyze this conversation context:" in message_content:
                # This is context analysis
                return '{"conversation_type": "new", "emotional_progression": "stable", "key_themes": ["anxiety"], "user_preferences": [], "suggested_focus": "support", "memory_cues": []}'

            elif "Manage memory for this interaction:" in message_content:
                # This is memory management
                return '{"to_remember": [], "to_forget": [], "memory_summary": "User expressed anxiety", "privacy_notes": ""}'

            else:
                # Default response for strategy-based generation
                return "I understand you're going through a difficult time. I'm here to listen and support you."

        mock_provider.generate_response = mock_generate_response
        return mock_provider

    @pytest.fixture
    def validator(self, mock_llm_provider):
        """Create a LangGraph validator with realistic mock responses."""
        return LangGraphQueryValidator(llm_provider=mock_llm_provider)

    @pytest.mark.asyncio
    async def test_complete_workflow_execution(self, validator):
        """Test complete workflow execution from start to finish."""
        query = "I'm feeling really anxious today"
        user_id = "test_user_123"
        conversation_history = [
            {"role": "user", "text": "Hello", "timestamp": datetime.now()},
            {"role": "assistant", "text": "Hi there!", "timestamp": datetime.now()}
        ]

        result = await validator.validate_query(query, user_id, conversation_history)

        # Verify result structure
        assert isinstance(result, dict)
        assert "query_type" in result
        assert "confidence" in result
        assert "routing_decision" in result
        assert "final_response" in result

        # Verify specific values
        assert result["query_type"] == "mental_support"
        assert result["confidence"] == 0.9
        assert result["routing_decision"] == "mental_health_support"

    @pytest.mark.asyncio
    async def test_crisis_workflow(self, validator):
        """Test workflow with crisis detection."""
        crisis_query = "I want to kill myself right now"

        result = await validator.validate_query(crisis_query)

        assert result["is_crisis"] == True  # Crisis correctly detected
        assert "crisis_severity" in result
        assert result["routing_decision"] == "crisis_intervention"

    @pytest.mark.asyncio
    async def test_random_question_workflow(self, validator):
        """Test workflow with random question."""
        random_query = "What's the weather like today?"

        result = await validator.validate_query(random_query)

        assert result["query_type"] == "random_question"
        assert result["routing_decision"] == "random_question_filtered"

    @pytest.mark.asyncio
    async def test_unclear_query_workflow(self, validator):
        """Test workflow with unclear query."""
        unclear_query = "xyz"

        result = await validator.validate_query(unclear_query)

        assert result["query_type"] == "unclear"
        assert result["routing_decision"] == "clarification_needed"

    @pytest.mark.asyncio
    async def test_workflow_with_conversation_history(self, validator):
        """Test workflow with conversation history."""
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
    async def test_workflow_error_handling(self):
        """Test workflow error handling."""
        # Create validator with None provider to test error handling
        validator = LangGraphQueryValidator(llm_provider=None)

        result = await validator.validate_query("Test query")

        assert result["query_type"] == "unclear"
        assert len(result["errors"]) > 0
        assert result["requires_human_intervention"]

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
        assert routing_decision == "standard_processing"

        # Test final response generation
        final_response = validator._generate_final_response(initial_state)
        assert isinstance(final_response, str)
        assert len(final_response) > 0

    @pytest.mark.asyncio
    async def test_workflow_performance(self, validator, benchmark_config):
        """Test workflow performance."""
        import time

        start_time = time.time()
        result = await validator.validate_query("Test query")
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within reasonable time (mocked, so very fast)
        assert execution_time < 1.0  # Less than 1 second
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_workflow_with_different_emotions(self, validator):
        """Test workflow with different emotional states."""
        test_cases = [
            ("I'm feeling sad", "mental_support"),
            ("I'm so angry right now", "mental_support"),
            ("I'm stressed about work", "mental_support"),
            ("I'm happy today", "mental_support"),  # Still mental support
            ("What is Python?", "random_question")  # Use a clearer technical question
        ]

        for query, expected_type in test_cases:
            result = await validator.validate_query(query)
            assert result["query_type"] == expected_type

    @pytest.mark.asyncio
    async def test_workflow_memory_management(self, validator):
        """Test memory management in workflow."""
        query = "I'm feeling anxious about my family"
        conversation_history = [
            {"role": "user", "text": "I need help", "timestamp": datetime.now()},
            {"role": "assistant", "text": "I'm here to help", "timestamp": datetime.now()}
        ]

        result = await validator.validate_query(query, conversation_history=conversation_history)

        assert "query_type" in result
        assert result["query_type"] == "mental_support"
        # Memory management should be included in the workflow


class TestLangGraphWorkflowIntegration:
    """Full integration tests for LangGraph workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_user_journey(self, mock_llm_provider):
        """Test complete user journey through the system."""
        validator = LangGraphQueryValidator(llm_provider=mock_llm_provider)

        # Simulate a user conversation
        conversation_flow = [
            "I'm feeling really anxious today",
            "Yes, it's been getting worse lately",
            "I think I need some help",
            "Thank you for listening"
        ]

        for i, query in enumerate(conversation_flow):
            result = await validator.validate_query(query)

            assert isinstance(result, dict)
            assert "query_type" in result
            assert "confidence" in result
            assert "final_response" in result

            # First query should be mental support
            if i == 0:
                assert result["query_type"] == "mental_support"

    @pytest.mark.asyncio
    async def test_workflow_with_escalation(self, mock_llm_provider):
        """Test workflow when escalation is needed."""
        validator = LangGraphQueryValidator(llm_provider=mock_llm_provider)

        # Mock crisis response
        async def mock_crisis_response(messages):
            if "crisis" in str(messages).lower():
                return '{"is_crisis": true, "confidence": 0.95, "crisis_indicators": ["suicidal ideation"], "severity": "critical", "recommended_action": "immediate_intervention"}'
            return "Mock response"

        mock_llm_provider.generate_response = mock_crisis_response

        crisis_query = "I don't want to live anymore"
        result = await validator.validate_query(crisis_query)

        assert result["is_crisis"] == True
        assert result["crisis_severity"] == "critical"
        assert result["routing_decision"] == "crisis_intervention"

    @pytest.mark.asyncio
    async def test_workflow_with_context_awareness(self, mock_llm_provider):
        """Test workflow with context awareness."""
        validator = LangGraphQueryValidator(llm_provider=mock_llm_provider)

        # Mock context-aware response
        async def mock_context_response(messages):
            if "context" in str(messages).lower():
                return '{"conversation_type": "ongoing", "emotional_progression": "worsening", "key_themes": ["anxiety", "depression"], "user_preferences": ["direct_communication"], "suggested_focus": "crisis_intervention", "memory_cues": ["User mentioned suicidal thoughts previously"]}'
            # For classification requests in context test
            elif "Query to classify:" in str(messages):
                return '{"query_type": "MENTAL_SUPPORT", "confidence": 0.9, "reasoning": "Mental health indicators detected", "keywords_found": ["anxious"], "is_crisis": false}'
            return "Mock response"

        mock_llm_provider.generate_response = mock_context_response

        conversation_history = [
            {"role": "user", "text": "I feel like ending it all", "timestamp": datetime.now()},
            {"role": "assistant", "text": "I'm concerned about you", "timestamp": datetime.now()}
        ]

        result = await validator.validate_query(
            "I'm still feeling the same way",
            conversation_history=conversation_history
        )

        assert result["query_type"] == "mental_support"
        # Context should influence the routing decision