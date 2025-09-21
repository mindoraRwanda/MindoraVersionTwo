#!/usr/bin/env python3
"""
Test script for the Query Validator Service
"""

from backend.app.services.query_validator import QueryValidatorService, QueryType

def test_query_validator():
    """Test the query validator with various queries"""
    validator = QueryValidatorService()

    # Test cases
    test_queries = [
        # Mental support queries
        ("I'm feeling really anxious today", QueryType.MENTAL_SUPPORT),
        ("I think I'm depressed and need help", QueryType.MENTAL_SUPPORT),
        ("Having panic attacks lately", QueryType.MENTAL_SUPPORT),
        ("I'm struggling with suicidal thoughts", QueryType.MENTAL_SUPPORT),
        ("Feeling overwhelmed with stress", QueryType.MENTAL_SUPPORT),

        # Random questions
        ("What's the weather like today?", QueryType.RANDOM_QUESTION),
        ("How do I install Python?", QueryType.RANDOM_QUESTION),
        ("What is the capital of France?", QueryType.RANDOM_QUESTION),
        ("How are you doing?", QueryType.RANDOM_QUESTION),
        ("Hey, what's up?", QueryType.RANDOM_QUESTION),

        # Unclear queries
        ("", QueryType.UNCLEAR),
        ("xyz", QueryType.UNCLEAR),
        ("maybe", QueryType.UNCLEAR),
    ]

    print("🧪 Testing Query Validator Service")
    print("=" * 50)

    correct_predictions = 0
    total_tests = len(test_queries)

    for query, expected_type in test_queries:
        result = validator.validate_query(query)


        # Check if prediction is correct
        is_correct = result.query_type == expected_type
        if is_correct:
            correct_predictions += 1

        # Print result
        status = "✅" if is_correct else "❌"
        print(f"{status} Query: '{query}'")
        print(f"   Expected: {expected_type.value}")
        print(f"   Got: {result.query_type.value}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Keywords: {', '.join(result.keywords_found)}")
        print(f"   Reasoning: {result.reasoning}")
        print()

    # Print summary
    accuracy = correct_predictions / total_tests * 100
    print("=" * 50)
    print(f"📊 Test Results: {correct_predictions}/{total_tests} correct ({accuracy:.1f}%)")

    # Test crisis detection
    print("\n🚨 Testing Crisis Detection:")
    crisis_queries = [
        "I'm thinking about killing myself",
        "I want to end it all right now",
        "Having thoughts of self harm",
        "This is an emergency, I need help"
    ]

    for query in crisis_queries:
        is_crisis, confidence = validator.is_crisis_query(query)
        print(f"Query: '{query}' -> Crisis: {is_crisis}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    test_query_validator()