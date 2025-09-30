"""
Integration tests for the chat router with query validation.

This module tests the integration between the chat router and the
LangGraph query validation system.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from fastapi.testclient import TestClient
from datetime import datetime

from backend.app.routers.chat_router import router, get_llm_service, get_query_validator
from backend.app.services.llm_service import LLMService
from backend.app.services.query_validator_langgraph import LangGraphQueryValidator
from tests import TestFixtures


class TestChatRouterIntegration:
    """Integration tests for chat router with query validation."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        mock_provider = AsyncMock()

        async def mock_generate_response(messages):
            return TestFixtures.mock_llm_response("Mock response from LLM")

        mock_provider.generate_response = mock_generate_response
        return mock_provider

    @pytest.fixture
    def mock_llm_service(self, mock_llm_provider):
        """Create a mock LLM service."""
        service = Mock(spec=LLMService)
        service.llm_provider = mock_llm_provider
        service.is_initialized = True
        service.generate_response = AsyncMock(return_value="Mock mental health response")
        return service

    @pytest.fixture
    def mock_query_validator(self, mock_llm_provider):
        """Create a mock query validator."""
        validator = Mock(spec=LangGraphQueryValidator)
        validator.validate_query = AsyncMock(return_value={
            "query_type": "mental_support",
            "confidence": 0.9,
            "reasoning": "Mental health query detected",
            "is_crisis": False,
            "routing_decision": "mental_health_support",
            "suggestions": ["Provide emotional support"],
            "final_response": "I understand you're going through a difficult time."
        })
        return validator

    @pytest.fixture
    def client(self, mock_llm_service, mock_query_validator):
        """Create a test client with mocked dependencies."""
        from fastapi import FastAPI

        app = FastAPI()

        # Override dependencies
        app.dependency_overrides[get_llm_service] = lambda: mock_llm_service
        app.dependency_overrides[get_query_validator] = lambda: mock_query_validator

        app.include_router(router)

        return TestClient(app)

    @pytest.mark.asyncio
    async def test_chat_with_mental_health_query(self, client, mock_query_validator):
        """Test chat endpoint with mental health query."""
        # Setup mock response
        mock_query_validator.validate_query.return_value = {
            "query_type": "mental_support",
            "confidence": 0.9,
            "reasoning": "Mental health indicators detected",
            "is_crisis": False,
            "routing_decision": "mental_health_support",
            "suggestions": ["Provide emotional support"],
            "final_response": "I understand you're going through a difficult time."
        }

        response = client.post(
            "/api/chat",
            json={"message": "I'm feeling really anxious today"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "response" in data
        assert "timestamp" in data
        assert "conversation_id" in data
        assert data["response"] == "I understand you're going through a difficult time."

    @pytest.mark.asyncio
    async def test_chat_with_random_question(self, client, mock_query_validator):
        """Test chat endpoint with random question."""
        # Setup mock response for random question
        mock_query_validator.validate_query.return_value = {
            "query_type": "random_question",
            "confidence": 0.8,
            "reasoning": "General question detected",
            "is_crisis": False,
            "routing_decision": "general_assistance",
            "suggestions": ["Provide general information"],
            "final_response": "I see you have a question about weather."
        }

        response = client.post(
            "/api/chat",
            json={"message": "What's the weather like today?"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "response" in data
        assert "I see you have a question" in data["response"]

    @pytest.mark.asyncio
    async def test_chat_with_crisis_query(self, client, mock_query_validator):
        """Test chat endpoint with crisis query."""
        # Setup mock response for crisis
        mock_query_validator.validate_query.return_value = {
            "query_type": "crisis",
            "confidence": 0.95,
            "reasoning": "Crisis indicators detected",
            "is_crisis": True,
            "crisis_severity": "critical",
            "routing_decision": "crisis_intervention",
            "suggestions": ["Immediate professional intervention"],
            "final_response": "I detect this may be a crisis situation."
        }

        response = client.post(
            "/api/chat",
            json={"message": "I want to end it all"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "crisis situation" in data["response"]
        assert "112" in data["response"]  # Emergency number
        assert "114" in data["response"]  # Mental health helpline

    @pytest.mark.asyncio
    async def test_chat_with_unclear_query(self, client, mock_query_validator):
        """Test chat endpoint with unclear query."""
        # Setup mock response for unclear query
        mock_query_validator.validate_query.return_value = {
            "query_type": "unclear",
            "confidence": 0.5,
            "reasoning": "Query unclear",
            "is_crisis": False,
            "routing_decision": "clarification_needed",
            "suggestions": ["Ask for clarification"],
            "final_response": "I'm not sure I understand your query."
        }

        response = client.post(
            "/api/chat",
            json={"message": "xyz"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "not sure I understand" in data["response"]
        assert "clarify" in data["response"]

    @pytest.mark.asyncio
    async def test_validate_query_endpoint(self, client, mock_query_validator):
        """Test the validate-query endpoint."""
        # Setup mock response
        mock_query_validator.validate_query.return_value = {
            "query_type": "mental_support",
            "confidence": 0.9,
            "reasoning": "Mental health query detected",
            "is_crisis": False,
            "routing_decision": "mental_health_support",
            "suggestions": ["Provide emotional support"],
            "final_response": "I understand you're going through a difficult time."
        }

        response = client.post(
            "/api/validate-query",
            json={"message": "I'm feeling anxious"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "I'm feeling anxious"
        assert data["validation"]["query_type"] == "mental_support"
        assert data["is_mental_health_related"] == True
        assert data["should_process"] == True
        assert data["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_chat_stats_endpoint(self, client):
        """Test the chat stats endpoint."""
        response = client.get("/api/chat-stats")

        assert response.status_code == 200
        data = response.json()

        assert "total_conversations" in data
        assert "total_messages" in data
        assert "validation_stats" in data
        assert "system_status" in data
        assert data["system_status"] == "operational"

    @pytest.mark.asyncio
    async def test_conversation_with_validation_metadata(self, client, mock_query_validator):
        """Test that conversations include validation metadata."""
        # Setup mock response
        mock_query_validator.validate_query.return_value = {
            "query_type": "mental_support",
            "confidence": 0.9,
            "reasoning": "Mental health query detected",
            "is_crisis": False,
            "routing_decision": "mental_health_support",
            "suggestions": ["Provide emotional support"],
            "final_response": "I understand you're going through a difficult time."
        }

        response = client.post(
            "/api/chat",
            json={"message": "I'm feeling anxious"}
        )

        assert response.status_code == 200
        data = response.json()

        # The response should include conversation_id for follow-up
        assert "conversation_id" in data

        # Test follow-up message in same conversation
        conversation_id = data["conversation_id"]
        response2 = client.post(
            "/api/chat",
            json={
                "message": "It's getting worse",
                "conversation_id": conversation_id
            }
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["conversation_id"] == conversation_id

    @pytest.mark.asyncio
    async def test_error_handling_in_chat(self, client, mock_query_validator):
        """Test error handling when query validation fails."""
        # Setup mock to raise an exception
        mock_query_validator.validate_query.side_effect = Exception("Validation failed")

        response = client.post(
            "/api/chat",
            json={"message": "Test message"}
        )

        # Should still return 200 but with fallback response
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    @pytest.mark.asyncio
    async def test_query_validation_endpoint_error_handling(self, client, mock_query_validator):
        """Test error handling in validate-query endpoint."""
        # Setup mock to raise an exception
        mock_query_validator.validate_query.side_effect = Exception("Validation failed")

        response = client.post(
            "/api/validate-query",
            json={"message": "Test message"}
        )

        # Should return 500 error
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Validation failed" in data["detail"]


class TestChatRouterIntegrationWithRealisticScenarios:
    """Realistic integration test scenarios."""

    @pytest.mark.asyncio
    async def test_user_journey_mental_health_support(self, client, mock_query_validator):
        """Test a complete user journey for mental health support."""
        # First message - mental health query
        mock_query_validator.validate_query.return_value = {
            "query_type": "mental_support",
            "confidence": 0.9,
            "reasoning": "Mental health indicators detected",
            "is_crisis": False,
            "routing_decision": "mental_health_support",
            "suggestions": ["Provide emotional support"],
            "final_response": "I understand you're feeling anxious. I'm here to help."
        }

        response1 = client.post(
            "/api/chat",
            json={"message": "I've been feeling really anxious lately"}
        )

        assert response1.status_code == 200
        data1 = response1.json()
        conversation_id = data1["conversation_id"]

        # Follow-up message in same conversation
        response2 = client.post(
            "/api/chat",
            json={
                "message": "It's affecting my sleep",
                "conversation_id": conversation_id
            }
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["conversation_id"] == conversation_id

    @pytest.mark.asyncio
    async def test_user_journey_random_to_mental_health(self, client, mock_query_validator):
        """Test user journey from random question to mental health."""
        # First message - random question
        mock_query_validator.validate_query.return_value = {
            "query_type": "random_question",
            "confidence": 0.8,
            "reasoning": "General question detected",
            "is_crisis": False,
            "routing_decision": "general_assistance",
            "suggestions": ["Redirect to mental health"],
            "final_response": "I see you have a question about weather, but I'm here for mental health support."
        }

        response1 = client.post(
            "/api/chat",
            json={"message": "What's the weather like?"}
        )

        assert response1.status_code == 200
        data1 = response1.json()
        conversation_id = data1["conversation_id"]

        # Follow-up with mental health concern
        mock_query_validator.validate_query.return_value = {
            "query_type": "mental_support",
            "confidence": 0.9,
            "reasoning": "Mental health indicators detected",
            "is_crisis": False,
            "routing_decision": "mental_health_support",
            "suggestions": ["Provide emotional support"],
            "final_response": "I understand you're feeling anxious. I'm here to help."
        }

        response2 = client.post(
            "/api/chat",
            json={
                "message": "Actually, I'm feeling anxious about something",
                "conversation_id": conversation_id
            }
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["conversation_id"] == conversation_id
        assert "anxious" in data2["response"].lower()

    @pytest.mark.asyncio
    async def test_crisis_escalation_flow(self, client, mock_query_validator):
        """Test crisis escalation flow."""
        # Crisis query
        mock_query_validator.validate_query.return_value = {
            "query_type": "crisis",
            "confidence": 0.95,
            "reasoning": "Crisis indicators detected",
            "is_crisis": True,
            "crisis_severity": "critical",
            "routing_decision": "crisis_intervention",
            "suggestions": ["Immediate professional intervention"],
            "final_response": "I detect this may be a crisis situation. Please contact emergency services immediately."
        }

        response = client.post(
            "/api/chat",
            json={"message": "I don't want to live anymore"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should contain crisis intervention information
        assert "crisis situation" in data["response"]
        assert "112" in data["response"]  # Emergency number
        assert "114" in data["response"]  # Mental health helpline
        assert "Ndera" in data["response"]  # Hospital information