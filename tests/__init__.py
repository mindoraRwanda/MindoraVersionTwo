"""
Tests package for the Mindora application.

This package contains all tests organized by module and functionality.
"""

import pytest
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

# Test configuration
pytest_plugins = ["pytest_asyncio"]

# Global test configuration
TEST_CONFIG = {
    "mock_llm": True,
    "mock_database": True,
    "test_timeout": 30,
    "max_test_concurrency": 5
}

# Test utilities
class TestUtils:
    """Utility functions for testing."""

    @staticmethod
    def create_mock_user(user_id: str = "test_user") -> Dict[str, Any]:
        """Create a mock user for testing."""
        return {
            "user_id": user_id,
            "email": f"{user_id}@test.com",
            "created_at": datetime.now(),
            "is_active": True
        }

    @staticmethod
    def create_mock_conversation(user_id: str = "test_user") -> Dict[str, Any]:
        """Create a mock conversation for testing."""
        return {
            "user_id": user_id,
            "conversation_id": f"conv_{user_id}",
            "messages": [
                {"role": "user", "text": "Hello", "timestamp": datetime.now()},
                {"role": "assistant", "text": "Hi there!", "timestamp": datetime.now()}
            ],
            "created_at": datetime.now()
        }

    @staticmethod
    def create_mock_query(query_text: str = "I'm feeling anxious") -> Dict[str, Any]:
        """Create a mock query for testing."""
        return {
            "query": query_text,
            "user_id": "test_user",
            "timestamp": datetime.now(),
            "session_id": "test_session"
        }

    @staticmethod
    async def async_delay(seconds: float = 0.1) -> None:
        """Async delay for testing."""
        await asyncio.sleep(seconds)

# Test fixtures
class TestFixtures:
    """Common test fixtures."""

    @staticmethod
    def mock_llm_response(response_text: str = "Test response") -> Dict[str, Any]:
        """Create a mock LLM response."""
        return {
            "content": response_text,
            "model": "test-model",
            "timestamp": datetime.now(),
            "tokens_used": 10
        }

    @staticmethod
    def mock_crisis_detection(is_crisis: bool = False) -> Dict[str, Any]:
        """Create a mock crisis detection result."""
        return {
            "is_crisis": is_crisis,
            "confidence": 0.8,
            "severity": "medium",
            "indicators": ["anxious", "overwhelmed"] if is_crisis else []
        }

    @staticmethod
    def mock_emotion_analysis(emotion: str = "anxiety") -> Dict[str, Any]:
        """Create a mock emotion analysis result."""
        return {
            "emotion": emotion,
            "confidence": 0.85,
            "intensity": "medium",
            "related_emotions": ["stress", "worry"]
        }