"""
Unit tests for the stateful mental health pipeline.

This module tests the core functionality of the stateful LangGraph pipeline
including query validation, crisis detection, emotion detection, and response generation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from backend.app.services.stateful_pipeline import StatefulMentalHealthPipeline
from backend.app.services.pipeline_state import (
    StatefulPipelineState, QueryValidationResult, CrisisAssessment,
    EmotionDetection, QueryEvaluation, QueryType, CrisisSeverity, ResponseStrategy
)
from backend.app.services.pipeline_nodes import (
    QueryValidationNode, CrisisDetectionNode, EmotionDetectionNode,
    QueryEvaluationNode, EmpathyNode, ElaborationNode
)


class TestStatefulPipeline:
    """Test suite for the stateful mental health pipeline."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider for testing."""
        mock_provider = AsyncMock()
        mock_provider.agenerate.return_value = "Mock LLM response"
        return mock_provider

    @pytest.fixture
    def pipeline(self, mock_llm_provider):
        """Create a pipeline instance for testing."""
        return StatefulMentalHealthPipeline(llm_provider=mock_llm_provider)

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, mock_llm_provider):
        """Test pipeline initialization."""
        pipeline = StatefulMentalHealthPipeline(llm_provider=mock_llm_provider)
        assert pipeline.llm_provider == mock_llm_provider
        assert pipeline.graph is not None

    @pytest.mark.asyncio
    async def test_process_query_basic(self, pipeline):
        """Test basic query processing."""
        query = "I'm feeling anxious about my exams"
        user_id = "test_user"
        
        result = await pipeline.process_query(
            query=query,
            user_id=user_id,
            conversation_history=[]
        )
        
        assert "response" in result
        assert "response_confidence" in result
        assert "processing_metadata" in result
        assert "query_validation" in result
        assert "crisis_assessment" in result
        assert "emotion_detection" in result
        assert "query_evaluation" in result

    @pytest.mark.asyncio
    async def test_process_query_with_history(self, pipeline):
        """Test query processing with conversation history."""
        query = "I'm still feeling anxious"
        user_id = "test_user"
        conversation_history = [
            {"role": "user", "text": "I'm feeling anxious about my exams"},
            {"role": "assistant", "text": "I understand you're feeling anxious. Can you tell me more?"}
        ]
        
        result = await pipeline.process_query(
            query=query,
            user_id=user_id,
            conversation_history=conversation_history
        )
        
        assert "response" in result
        assert len(result["processing_metadata"]) > 0

    @pytest.mark.asyncio
    async def test_pipeline_fallback_on_error(self, mock_llm_provider):
        """Test pipeline fallback behavior on errors."""
        # Make LLM provider raise an exception
        mock_llm_provider.agenerate.side_effect = Exception("LLM error")
        
        pipeline = StatefulMentalHealthPipeline(llm_provider=mock_llm_provider)
        
        result = await pipeline.process_query(
            query="Test query",
            user_id="test_user"
        )
        
        assert result["response"] == "I'm here to support you. How can I help you today?"
        assert result["response_confidence"] == 0.0
        assert "LLM error" in result["response_reason"]


class TestQueryValidationNode:
    """Test suite for the query validation node."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider for testing."""
        mock_provider = AsyncMock()
        mock_provider.agenerate.return_value = """
        {
            "confidence": 0.8,
            "keywords": ["anxiety", "exams"],
            "reasoning": "Query contains mental health indicators",
            "is_random": false,
            "query_type": "mental_health"
        }
        """
        return mock_provider

    @pytest.fixture
    def validation_node(self, mock_llm_provider):
        """Create a query validation node for testing."""
        return QueryValidationNode(llm_provider=mock_llm_provider)

    @pytest.mark.asyncio
    async def test_query_validation_mental_health(self, validation_node):
        """Test query validation for mental health queries."""
        state = {
            "user_query": "I'm feeling anxious about my exams",
            "processing_metadata": [],
            "processing_steps_completed": [],
            "llm_calls_made": 0,
            "errors": []
        }
        
        result = await validation_node.execute(state)
        
        assert "query_validation" in result
        assert result["query_validation"].query_confidence == 0.8
        assert result["query_validation"].is_random == False
        assert result["query_validation"].query_type == QueryType.MENTAL_HEALTH
        assert "query_validation" in result["processing_steps_completed"]

    @pytest.mark.asyncio
    async def test_query_validation_random_query(self, validation_node):
        """Test query validation for random queries."""
        # Mock LLM to return random query classification
        validation_node.llm_provider.agenerate.return_value = """
        {
            "confidence": 0.1,
            "keywords": [],
            "reasoning": "Query is about general topics",
            "is_random": true,
            "query_type": "random"
        }
        """
        
        state = {
            "user_query": "What's the weather like today?",
            "processing_metadata": [],
            "processing_steps_completed": [],
            "llm_calls_made": 0,
            "errors": []
        }
        
        result = await validation_node.execute(state)
        
        assert result["query_validation"].is_random == True
        assert result["query_validation"].query_type == QueryType.RANDOM


class TestCrisisDetectionNode:
    """Test suite for the crisis detection node."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider for testing."""
        mock_provider = AsyncMock()
        mock_provider.agenerate.return_value = """
        {
            "confidence": 0.1,
            "keywords": [],
            "reasoning": "No crisis indicators detected",
            "severity": "none"
        }
        """
        return mock_provider

    @pytest.fixture
    def crisis_node(self, mock_llm_provider):
        """Create a crisis detection node for testing."""
        return CrisisDetectionNode(llm_provider=mock_llm_provider)

    @pytest.mark.asyncio
    async def test_crisis_detection_no_crisis(self, crisis_node):
        """Test crisis detection for non-crisis queries."""
        state = {
            "user_query": "I'm feeling a bit stressed",
            "processing_metadata": [],
            "processing_steps_completed": [],
            "llm_calls_made": 0,
            "errors": []
        }

        result = await crisis_node.execute(state)

        assert "crisis_assessment" in result
        assert result["crisis_assessment"].crisis_severity == CrisisSeverity.NONE
        assert result["crisis_assessment"].crisis_confidence == 0.1
    @pytest.mark.asyncio
    async def test_crisis_detection_high_crisis(self, crisis_node):
        """Test crisis detection for high-risk queries."""
        # Mock LLM to return high crisis classification
        crisis_node.llm_provider.agenerate.return_value = """
        {
            "confidence": 0.8,
            "keywords": ["suicide", "end", "life"],
            "reasoning": "Query contains severe crisis indicators",
            "severity": "severe"
        }
        """
        
        state = {
            "user_query": "I want to end my life",
            "processing_metadata": [],
            "processing_steps_completed": [],
            "llm_calls_made": 0,
            "errors": []
        }
        
        result = await crisis_node.execute(state)
        
        assert result["crisis_assessment"].crisis_severity == CrisisSeverity.SEVERE
        assert result["crisis_assessment"].crisis_confidence == 0.8


class TestEmotionDetectionNode:
    """Test suite for the emotion detection node."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider for testing."""
        mock_provider = AsyncMock()
        mock_provider.agenerate.return_value = """
        {
            "emotions": {"anxiety": 0.7, "neutral": 0.3},
            "keywords": ["anxious", "worried"],
            "reasoning": "Query shows anxiety and worry",
            "selected_emotion": "anxiety",
            "confidence": 0.7
        }
        """
        return mock_provider

    @pytest.fixture
    def emotion_node(self, mock_llm_provider):
        """Create an emotion detection node for testing."""
        return EmotionDetectionNode(llm_provider=mock_llm_provider)

    @pytest.mark.asyncio
    async def test_emotion_detection(self, emotion_node):
        """Test emotion detection functionality."""
        state = {
            "user_query": "I'm feeling really anxious about everything",
            "processing_metadata": [],
            "processing_steps_completed": [],
            "llm_calls_made": 0,
            "errors": []
        }
        
        result = await emotion_node.execute(state)
        
        assert "emotion_detection" in result
        assert result["emotion_detection"].selected_emotion == "anxiety"
        assert result["emotion_detection"].emotion_confidence == 0.7
        assert "anxiety" in result["emotion_detection"].emotions


class TestQueryEvaluationNode:
    """Test suite for the query evaluation node."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider for testing."""
        mock_provider = AsyncMock()
        mock_provider.agenerate.return_value = """
        {
            "confidence": 0.8,
            "reasoning": "User needs emotional support",
            "keywords": ["support", "help"],
            "strategy": "GIVE_EMPATHY"
        }
        """
        return mock_provider

    @pytest.fixture
    def evaluation_node(self, mock_llm_provider):
        """Create a query evaluation node for testing."""
        return QueryEvaluationNode(llm_provider=mock_llm_provider)

    @pytest.mark.asyncio
    async def test_query_evaluation(self, evaluation_node):
        """Test query evaluation and strategy selection."""
        state = {
            "user_query": "I'm feeling really down and need support",
            "crisis_assessment": CrisisAssessment(
                crisis_confidence=0.1,
                crisis_keywords=[],
                crisis_reason="No crisis indicators",
                crisis_severity=CrisisSeverity.NONE
            ),
            "emotion_detection": EmotionDetection(
                emotions={"sadness": 0.8},
                emotion_keywords=["down", "sad"],
                emotion_reason="Shows sadness",
                selected_emotion="sadness",
                emotion_confidence=0.8
            ),
            "processing_metadata": [],
            "processing_steps_completed": [],
            "llm_calls_made": 0,
            "errors": []
        }
        
        result = await evaluation_node.execute(state)
        
        assert "query_evaluation" in result
        assert result["query_evaluation"].evaluation_type == ResponseStrategy.GIVE_EMPATHY
        assert result["query_evaluation"].evaluation_confidence == 0.8


class TestResponseNodes:
    """Test suite for response generation nodes."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider for testing."""
        mock_provider = AsyncMock()
        mock_provider.agenerate.return_value = "I understand you're going through a difficult time. I'm here to support you."
        return mock_provider

    @pytest.mark.asyncio
    async def test_empathy_node(self, mock_llm_provider):
        """Test empathy response generation."""
        empathy_node = EmpathyNode(llm_provider=mock_llm_provider)
        
        state = {
            "user_query": "I'm feeling really sad",
            "emotion_detection": EmotionDetection(
                emotions={"sadness": 0.8},
                emotion_keywords=["sad"],
                emotion_reason="Shows sadness",
                selected_emotion="sadness",
                emotion_confidence=0.8
            ),
            "processing_metadata": [],
            "processing_steps_completed": [],
            "llm_calls_made": 0,
            "errors": []
        }
        
        result = await empathy_node.execute(state)
        
        assert "generated_content" in result
        assert result["response_confidence"] == 0.8
        assert "empathy_response" in result["processing_steps_completed"]

    @pytest.mark.asyncio
    async def test_elaboration_node(self, mock_llm_provider):
        """Test elaboration response generation."""
        elaboration_node = ElaborationNode(llm_provider=mock_llm_provider)
        
        state = {
            "user_query": "I'm having some issues",
            "processing_metadata": [],
            "processing_steps_completed": [],
            "llm_calls_made": 0,
            "errors": []
        }
        
        result = await elaboration_node.execute(state)
        
        assert "generated_content" in result
        assert result["response_confidence"] == 0.7
        assert "elaboration_response" in result["processing_steps_completed"]


if __name__ == "__main__":
    pytest.main([__file__])
