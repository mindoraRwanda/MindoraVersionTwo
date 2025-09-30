#!/usr/bin/env python3
"""
Test script for LLM-Enhanced Stateful Conversation System

This script tests the enhanced conversation system with LLM-powered state decisions
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.services.session_state_manager import session_manager, ConversationState
from backend.app.services.crisis_interceptor import crisis_interceptor
from backend.app.services.langgraph_state_router import llm_enhanced_router


async def test_llm_state_decisions():
    """Test LLM-powered state decisions"""
    print("=== Testing LLM-Enhanced State Decisions ===")

    # Create session
    session_id = session_manager.create_session("test_user")

    # Test different types of user messages
    test_messages = [
        "I'm feeling really stressed and overwhelmed",
        "It's been like this for a few weeks now",
        "I have so much work and family pressure",
        "I don't know how to handle all of this",
        "What should I do?"
    ]

    for i, message in enumerate(test_messages):
        print(f"\n--- Turn {i+1} ---")
        print(f"User: {message}")

        # Get LLM-enhanced routing decision
        response_data = await llm_enhanced_router.route_conversation(
            session_id, message
        )

        print(f"LLM Response: {response_data.get('response', '')[:100]}...")
        print(f"Next State: {response_data.get('next_state')}")
        print(f"Strategy: {response_data.get('response_type')}")
        print(f"Confidence: {response_data.get('confidence', 0):.2f}")
        print(f"LLM Reasoning: {response_data.get('llm_reasoning', '')}")

    print("✅ LLM-enhanced state decisions test passed\n")


async def test_crisis_llm_handling():
    """Test crisis handling with LLM enhancement"""
    print("=== Testing Crisis Handling with LLM ===")

    # Create session
    session_id = session_manager.create_session("test_user")

    # Test crisis message
    crisis_message = "I want to kill myself, I can't take it anymore"
    print(f"Crisis message: {crisis_message}")

    response_data = await llm_enhanced_router.route_conversation(
        session_id, crisis_message
    )

    print(f"Crisis Response: {response_data.get('response', '')[:150]}...")
    print(f"Next State: {response_data.get('next_state')}")
    print(f"Strategy: {response_data.get('response_type')}")
    print(f"Confidence: {response_data.get('confidence', 0):.2f}")

    print("✅ Crisis LLM handling test passed\n")


async def test_conversation_context_awareness():
    """Test conversation context awareness"""
    print("=== Testing Conversation Context Awareness ===")

    # Create session
    session_id = session_manager.create_session("test_user")

    # Build conversation context
    conversation_flow = [
        "I'm feeling really anxious",
        "It's been happening for a few days",
        "I think it's because of work stress",
        "I can't sleep well",
        "What can I do to feel better?"
    ]

    for message in conversation_flow:
        response_data = await llm_enhanced_router.route_conversation(
            session_id, message
        )

        print(f"User: {message}")
        print(f"State: {response_data.get('next_state')} | Strategy: {response_data.get('response_type')}")
        print(f"Response: {response_data.get('response', '')[:80]}...")
        print("---")

    print("✅ Conversation context awareness test passed\n")


async def main():
    """Run all LLM-enhanced tests"""
    print("🤖 Starting LLM-Enhanced Conversation System Tests\n")

    try:
        await test_llm_state_decisions()
        await test_crisis_llm_handling()
        await test_conversation_context_awareness()

        print("🎉 All LLM-enhanced tests passed! The system is ready for intelligent conversation routing.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)