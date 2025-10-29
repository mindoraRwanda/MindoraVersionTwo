"""
Pytest configuration and fixtures for the Mindora application tests.

This file contains global fixtures and configuration for all tests.
"""

import pytest
import pytest_asyncio
import asyncio
from typing import AsyncGenerator, Dict, Any, List
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from backend.app.services.llm_service import LLMService
from tests import TestUtils, TestFixtures


# Global test configuration
@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        "mock_llm": True,
        "mock_database": True,
        "test_timeout": 30,
        "max_test_concurrency": 5,
        "test_user_id": "test_user_123",
        "test_session_id": "test_session_456"
    }


# Mock LLM Provider fixture
@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    mock_provider = AsyncMock()

    # Mock response for classification
    async def mock_generate_response(messages):
        # Check if this is a classification request (contains "Query to classify")
        if messages and len(messages) > 1 and "Query to classify" in str(messages[1]):
            # Return JSON response for classification
            return '''{
                "query_type": "RANDOM_QUESTION",
                "confidence": 0.9,
                "reasoning": "Technical question detected",
                "keywords_found": ["python", "programming"],
                "is_crisis": false
            }'''
        else:
            # Return plain text for other requests
            return "Mock response from LLM service"

    mock_provider.generate_response = mock_generate_response
    return mock_provider


# Mock LLM Service fixture
@pytest.fixture
async def mock_llm_service():
    """Mock LLM service for testing."""
    service = AsyncMock(spec=LLMService)

    # Mock the llm_provider attribute
    service.llm_provider = AsyncMock()
    service.is_initialized = True

    # Mock generate_response method
    async def mock_generate_response(messages):
        # Check if this is a classification request (contains "Query to classify")
        if messages and len(messages) > 1 and "Query to classify" in str(messages[1]):
            # Return JSON response for classification
            return '''{
                "query_type": "RANDOM_QUESTION",
                "confidence": 0.9,
                "reasoning": "Technical question detected",
                "keywords_found": ["python", "programming"],
                "is_crisis": false
            }'''
        else:
            # Return plain text for other requests
            return "Mock response from LLM service"

    service.llm_provider.generate_response = mock_generate_response

    return service


# Test user fixture
@pytest.fixture
def test_user():
    """Create a test user for testing."""
    return TestUtils.create_mock_user("test_user_123")


# Test conversation fixture
@pytest.fixture
def test_conversation():
    """Create a test conversation for testing."""
    return TestUtils.create_mock_conversation("test_user_123")


# Test query fixture
@pytest.fixture
def test_query():
    """Create a test query for testing."""
    return TestUtils.create_mock_query("I'm feeling anxious today")


# Stateful pipeline fixture
@pytest.fixture
async def stateful_pipeline(mock_llm_provider):
    """Create a stateful pipeline with mock LLM provider."""
    from backend.app.services.stateful_pipeline import StatefulMentalHealthPipeline
    pipeline = StatefulMentalHealthPipeline(llm_provider=mock_llm_provider)
    return pipeline


# Database mock fixture
@pytest.fixture
def mock_database():
    """Mock database session for testing."""
    mock_db = Mock()
    mock_session = Mock()

    # Mock query methods
    mock_session.query.return_value.filter_by.return_value.first.return_value = None
    mock_session.query.return_value.filter_by.return_value.order_by.return_value.first.return_value = None
    mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []

    mock_db.return_value.__enter__.return_value = mock_session
    mock_db.return_value.__exit__.return_value = None

    return mock_db


# Async test utilities
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test data fixtures
@pytest.fixture
def sample_mental_health_queries():
    """Sample mental health queries for testing."""
    return [
        "I'm feeling really anxious today",
        "I think I'm depressed and need help",
        "Having panic attacks lately",
        "I'm struggling with stress",
        "Feeling overwhelmed with everything"
    ]


@pytest.fixture
def sample_random_queries():
    """Sample random questions for testing."""
    return [
        "What's the weather like today?",
        "How do I install Python?",
        "What is the capital of France?",
        "How are you doing?",
        "What's your favorite color?"
    ]


@pytest.fixture
def sample_crisis_queries():
    """Sample crisis queries for testing."""
    return [
        "I'm thinking about killing myself",
        "I want to end it all right now",
        "Having thoughts of self harm",
        "This is an emergency, I need help"
    ]


# Performance testing fixtures
@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarking."""
    return {
        "min_rounds": 5,
        "max_time": 1.0,
        "warmup": True,
        "warmup_rounds": 2
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Add any cleanup code here if needed


# Pytest hooks
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "llm: marks tests that require LLM")
    config.addinivalue_line("markers", "database: marks tests that require database")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add integration marker to tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add llm marker to tests that use LLM
        if "llm" in item.nodeid.lower() or "langgraph" in item.nodeid.lower():
            item.add_marker(pytest.mark.llm)

        # Add database marker to tests that use database
        if "database" in item.nodeid.lower() or "db" in item.nodeid.lower():
            item.add_marker(pytest.mark.database)