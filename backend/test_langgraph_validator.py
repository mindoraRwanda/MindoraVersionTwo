#!/usr/bin/env python3
"""
Test script for the LangGraph Query Validator Service
"""

import asyncio
from backend.app.services.query_validator_langgraph import LangGraphQueryValidator
from backend.app.services.llm_service import LLMService

async def test_langgraph_validator():
    """Test the LangGraph query validator with various queries"""
    print("🧪 Testing LangGraph Query Validator Service")
    print("=" * 60)

    # Initialize LLM service (this would normally be done in main.py)
    llm_service = LLMService(use_vllm=False, provider_name="huggingface", model_name="HuggingFaceTB/SmolLM3-3B")

    # Initialize LangGraph validator with LLM provider
    validator = LangGraphQueryValidator(llm_provider=llm_service.llm_provider)

    # Test cases
    test_queries = [
        # Mental support queries
        ("I'm feeling really anxious today", "MENTAL_SUPPORT"),
        ("I think I'm depressed and need help", "MENTAL_SUPPORT"),
        ("Having panic attacks lately", "MENTAL_SUPPORT"),
        ("I'm struggling with suicidal thoughts", "CRISIS"),

        # Random questions
        ("What's the weather like today?", "RANDOM_QUESTION"),
        ("How do I install Python?", "RANDOM_QUESTION"),
        ("What is the capital of France?", "RANDOM_QUESTION"),
        ("How are you doing?", "RANDOM_QUESTION"),

        # Unclear queries
        ("", "UNCLEAR"),
        ("xyz", "UNCLEAR"),
        ("maybe", "UNCLEAR"),
    ]

    print("Note: This test requires a running LLM service.")
    print("If LLM is not available, it will use fallback responses.\n")

    for query, expected_type in test_queries:
        print(f"Query: '{query}'")
        print(f"Expected: {expected_type}")

        try:
            result = await validator.validate_query(query)

            print(f"Got: {result['query_type']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Crisis: {result['is_crisis']}")
            print(f"Routing: {result['routing_decision']}")
            print(f"Response: {result['final_response'][:100]}...")
            print()

        except Exception as e:
            print(f"Error: {e}")
            print()

    print("=" * 60)
    print("✅ LangGraph Query Validator Test Complete")

    # Test crisis detection specifically
    print("\n🚨 Testing Crisis Detection:")
    crisis_queries = [
        "I'm thinking about killing myself",
        "I want to end it all right now",
        "Having thoughts of self harm",
        "This is an emergency, I need help"
    ]

    for query in crisis_queries:
        try:
            result = await validator.validate_query(query)
            print(f"Query: '{query}' -> Crisis: {result['is_crisis']}, Severity: {result['crisis_severity']}")
        except Exception as e:
            print(f"Query: '{query}' -> Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_langgraph_validator())