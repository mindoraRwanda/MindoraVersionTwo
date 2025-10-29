"""
Unit tests for the prompts package.

This module contains tests for all prompt modules and their functionality.
"""

import pytest
from unittest.mock import Mock, patch

from backend.app.prompts.system_prompts import SystemPrompts
from backend.app.prompts.query_classification_prompts import QueryClassificationPrompts
from backend.app.prompts.safety_prompts import SafetyPrompts
from backend.app.prompts.cultural_context_prompts import CulturalContextPrompts
from backend.app.prompts.response_approach_prompts import ResponseApproachPrompts


class TestSystemPrompts:
    """Tests for SystemPrompts class."""

    def test_get_main_system_prompt(self):
        """Test main system prompt generation."""
        prompt = SystemPrompts.get_main_system_prompt(
            context="Test context",
            emotion="anxiety",
            validation="Test validation",
            support_offering="Test support",
            crisis_helpline="Test helpline",
            emergency="Test emergency"
        )

        assert isinstance(prompt, str)
        assert "Mindora Chat Companion" in prompt
        assert "Test context" in prompt
        assert "anxiety" in prompt
        assert "Test helpline" in prompt

    def test_get_fallback_response(self):
        """Test fallback response."""
        response = SystemPrompts.get_fallback_response()

        assert isinstance(response, str)
        assert "difficult time" in response
        assert "grounding techniques" in response

    def test_get_grounding_exercise(self):
        """Test grounding exercise."""
        exercise = SystemPrompts.get_grounding_exercise()

        assert isinstance(exercise, str)
        assert "ground ourselves" in exercise
        assert "breathe" in exercise.lower()

    def test_get_error_messages(self):
        """Test error messages."""
        messages = SystemPrompts.get_error_messages()

        assert isinstance(messages, dict)
        assert "model_not_initialized" in messages
        assert "ollama_not_running" in messages


class TestQueryClassificationPrompts:
    """Tests for QueryClassificationPrompts class."""

    def test_get_query_classification_prompt(self):
        """Test query classification prompt."""
        prompt = QueryClassificationPrompts.get_query_classification_prompt()

        assert isinstance(prompt, str)
        assert "query classifier" in prompt.lower()
        assert "MENTAL_SUPPORT" in prompt
        assert "RANDOM_QUESTION" in prompt
        assert "CRISIS" in prompt

    def test_get_crisis_detection_prompt(self):
        """Test crisis detection prompt."""
        prompt = QueryClassificationPrompts.get_crisis_detection_prompt()

        assert isinstance(prompt, str)
        assert "crisis detection" in prompt.lower()
        assert "suicidal" in prompt.lower()
        assert "self-harm" in prompt.lower()

    def test_get_query_suggestions_prompt(self):
        """Test query suggestions prompt."""
        prompt = QueryClassificationPrompts.get_query_suggestions_prompt()

        assert isinstance(prompt, str)
        assert "query routing" in prompt.lower()
        assert "suggestions" in prompt.lower()

    def test_parse_classification_response(self):
        """Test parsing classification response."""
        # Test valid JSON response
        valid_response = '{"query_type": "MENTAL_SUPPORT", "confidence": 0.9, "reasoning": "Test"}'
        result = QueryClassificationPrompts.parse_classification_response(valid_response)

        assert isinstance(result, dict)
        assert result["query_type"] == "MENTAL_SUPPORT"
        assert result["confidence"] == 0.9

        # Test invalid response
        invalid_response = "Some random text without JSON"
        result = QueryClassificationPrompts.parse_classification_response(invalid_response)

        assert result["query_type"] == "UNCLEAR"
        assert result["confidence"] == 0.5


class TestSafetyPrompts:
    """Tests for SafetyPrompts class."""

    def test_get_safety_responses(self):
        """Test safety response templates."""
        from backend.app.services.llm_safety import SafetyManager
        response = SafetyManager.check_safety("I want to hurt myself")

        assert isinstance(response, str) or response is None
        assert "safety_and_boundary_check" in config
        assert "self_harm" in config
        assert "crisis" in config

    def test_get_crisis_keywords(self):
        """Test crisis keywords."""
        keywords = SafetyPrompts.get_crisis_keywords()

        assert isinstance(keywords, list)
        assert "suicide" in keywords
        assert "kill myself" in keywords
        assert len(keywords) > 5

    def test_get_safety_keyword_methods(self):
        """Test all safety keyword methods."""
        methods = [
            SafetyPrompts.get_substance_abuse_keywords,
            SafetyPrompts.get_self_injury_keywords,
            SafetyPrompts.get_illegal_content_keywords,
            SafetyPrompts.get_jailbreak_keywords,
            SafetyPrompts.get_inappropriate_relationship_keywords,
            SafetyPrompts.get_medical_advice_keywords,
            SafetyPrompts.get_mental_health_indicators
        ]

        for method in methods:
            keywords = method()
            assert isinstance(keywords, list)
            assert len(keywords) > 0

    def test_get_injection_patterns(self):
        """Test injection patterns."""
        patterns = SafetyPrompts.get_injection_patterns()

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert all(isinstance(pattern, str) for pattern in patterns)

    def test_get_unsafe_output_patterns(self):
        """Test unsafe output patterns."""
        patterns = SafetyPrompts.get_unsafe_output_patterns()

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert all(isinstance(pattern, str) for pattern in patterns)

    def test_get_simple_greetings(self):
        """Test simple greetings."""
        greetings = SafetyPrompts.get_simple_greetings()

        assert isinstance(greetings, list)
        assert "hello" in greetings
        assert "hi" in greetings


class TestCulturalContextPrompts:
    """Tests for CulturalContextPrompts class."""

    def test_get_rwanda_cultural_context(self):
        """Test Rwanda cultural context."""
        context = CulturalContextPrompts.get_rwanda_cultural_context()

        assert isinstance(context, dict)
        assert "ubuntu_philosophy" in context
        assert "family_support" in context
        assert "resilience_history" in context

    def test_get_rwanda_crisis_resources(self):
        """Test Rwanda crisis resources."""
        resources = CulturalContextPrompts.get_rwanda_crisis_resources()

        assert isinstance(resources, dict)
        assert "national_helpline" in resources
        assert "emergency" in resources
        assert "hospitals" in resources

    def test_get_emotion_responses(self):
        """Test emotion responses."""
        responses = CulturalContextPrompts.get_emotion_responses()

        assert isinstance(responses, dict)
        assert "sadness" in responses
        assert "anxiety" in responses
        assert "stress" in responses

        # Check structure of emotion response
        sadness_response = responses["sadness"]
        assert "tone" in sadness_response
        assert "validation" in sadness_response
        assert "exploration_question" in sadness_response

    def test_get_topic_adjustments(self):
        """Test topic adjustments."""
        adjustments = CulturalContextPrompts.get_topic_adjustments()

        assert isinstance(adjustments, dict)
        assert "school" in adjustments
        assert "family" in adjustments
        assert "work" in adjustments

    def test_get_cultural_integration_prompt(self):
        """Test cultural integration prompt."""
        prompt = CulturalContextPrompts.get_cultural_integration_prompt()

        assert isinstance(prompt, str)
        assert "culturally aware Rwandan" in prompt
        assert "Rwandan experience" in prompt
        assert "cultural wisdom" in prompt
        assert "gender" in prompt.lower()

    def test_get_resource_referral_prompt(self):
        """Test resource referral prompt."""
        prompt = CulturalContextPrompts.get_resource_referral_prompt()

        assert isinstance(prompt, str)
        assert "resource connector" in prompt.lower()
        assert "Rwanda" in prompt
        assert "helpline" in prompt.lower()


class TestResponseApproachPrompts:
    """Tests for ResponseApproachPrompts class."""

    def test_get_response_approach_prompt(self):
        """Test response approach prompt."""
        prompt = ResponseApproachPrompts.get_response_approach_prompt()

        assert isinstance(prompt, str)
        assert "therapeutic communication" in prompt.lower()
        assert "emotional states" in prompt.lower()
        assert "response approaches" in prompt.lower()

    def test_get_conversation_context_prompt(self):
        """Test conversation context prompt."""
        prompt = ResponseApproachPrompts.get_conversation_context_prompt()

        assert isinstance(prompt, str)
        assert "conversation context" in prompt.lower()
        assert "emotional progression" in prompt.lower()
        assert "topics discussed" in prompt.lower()

    def test_get_memory_management_prompt(self):
        """Test memory management prompt."""
        prompt = ResponseApproachPrompts.get_memory_management_prompt()

        assert isinstance(prompt, str)
        assert "memory management" in prompt.lower()
        assert "personal information" in prompt.lower()
        assert "privacy" in prompt.lower()


class TestPromptsIntegration:
    """Integration tests for the prompts package."""

    def test_all_prompts_are_strings(self):
        """Test that all prompts return strings."""
        # Test system prompts
        assert isinstance(SystemPrompts.get_main_system_prompt(), str)
        assert isinstance(SystemPrompts.get_fallback_response(), str)
        assert isinstance(SystemPrompts.get_grounding_exercise(), str)

        # Test classification prompts
        assert isinstance(QueryClassificationPrompts.get_query_classification_prompt(), str)
        assert isinstance(QueryClassificationPrompts.get_crisis_detection_prompt(), str)
        assert isinstance(QueryClassificationPrompts.get_query_suggestions_prompt(), str)

        # Test safety system
        from backend.app.services.llm_safety import SafetyManager
        assert SafetyManager.check_safety("test message") is None or isinstance(SafetyManager.check_safety("test message"), str)

        # Test cultural prompts
        assert isinstance(CulturalContextPrompts.get_cultural_integration_prompt(), str)
        assert isinstance(CulturalContextPrompts.get_resource_referral_prompt(), str)

        # Test response approach prompts
        assert isinstance(ResponseApproachPrompts.get_response_approach_prompt(), str)
        assert isinstance(ResponseApproachPrompts.get_conversation_context_prompt(), str)
        assert isinstance(ResponseApproachPrompts.get_memory_management_prompt(), str)

    def test_all_prompts_contain_expected_content(self):
        """Test that prompts contain expected keywords."""
        # System prompts
        system_prompt = SystemPrompts.get_main_system_prompt()
        assert "Mindora Chat Companion" in system_prompt
        assert "Rwanda" in system_prompt

        # Classification prompts
        classification_prompt = QueryClassificationPrompts.get_query_classification_prompt()
        assert "MENTAL_SUPPORT" in classification_prompt
        assert "RANDOM_QUESTION" in classification_prompt

        # Safety system
        from backend.app.services.llm_safety import SafetyManager
        safety_response = SafetyManager.get_safety_response("self_harm")
        assert safety_response is not None
        assert "crisis" in safety_response or "help" in safety_response

        # Cultural prompts
        cultural_prompt = CulturalContextPrompts.get_cultural_integration_prompt()
        assert "Ubuntu" in cultural_prompt
        assert "cultural" in cultural_prompt.lower()

    def test_prompt_formatting_with_parameters(self):
        """Test prompt formatting with different parameters."""
        # Test system prompt with different emotions
        anxiety_prompt = SystemPrompts.get_main_system_prompt(emotion="anxiety")
        sadness_prompt = SystemPrompts.get_main_system_prompt(emotion="sadness")

        assert "anxiety" in anxiety_prompt
        assert "sadness" in sadness_prompt

        # Test with different contexts
        context_prompt = SystemPrompts.get_main_system_prompt(context="Test context")
        assert "Test context" in context_prompt