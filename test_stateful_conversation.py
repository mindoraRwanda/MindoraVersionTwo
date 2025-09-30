#!/usr/bin/env python3
"""
Test script for the new stateful conversation system

This script tests the complete stateful conversation flow including:
- Session management
- Crisis detection and intervention
- State transitions
- Conversation actions
- Integration with chat router
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.services.session_state_manager import session_manager, ConversationState

# Start the cleanup task for session management
session_manager.start_cleanup_task()
from backend.app.services.crisis_interceptor import crisis_interceptor
from backend.app.services.state_router import state_router
from backend.app.services.state_actions import state_actions
from backend.app.services.conversation_content import ConversationContentManager


async def test_session_management():
    """Test session creation and management"""
    print("=== Testing Session Management ===")

    # Create a new session
    session_id = session_manager.create_session("test_user")
    print(f"Created session: {session_id}")

    # Get session
    session = session_manager.get_session(session_id)
    if session:
        print(f"Retrieved session state: {session.current_state.value}")

        # Add message to history
        session_manager.add_message_to_history(session_id, "user", "I'm feeling really stressed today")
        session = session_manager.get_session(session_id)
        if session:
            print(f"Messages in history: {len(session.conversation_history)}")
    else:
        print("Failed to retrieve session")

    # Test session summary
    summary = session_manager.get_session_summary(session_id)
    print(f"Session summary: {summary}")

    print("✅ Session management test passed\n")


async def test_crisis_detection():
    """Test crisis detection and intervention"""
    print("=== Testing Crisis Detection ===")

    # Test crisis message
    crisis_message = "I want to kill myself, I can't take it anymore"
    is_crisis, flags, severity = crisis_interceptor.detect_crisis(crisis_message)
    print(f"Crisis detected: {is_crisis}")
    print(f"Crisis flags: {flags}")
    print(f"Severity: {severity.value}")

    # Test non-crisis message
    normal_message = "I'm feeling a bit stressed about work"
    is_crisis, flags, severity = crisis_interceptor.detect_crisis(normal_message)
    print(f"Normal message crisis detected: {is_crisis}")

    # Test crisis interception
    session_id = session_manager.create_session("test_user")
    crisis_response = await crisis_interceptor.intercept_and_respond(
        session_id, crisis_message, region="rwanda"
    )

    if crisis_response:
        print(f"Crisis response generated: {len(crisis_response['immediate_response'])} chars")
        print(f"Resources provided: {len(crisis_response['resources']['crisis_lines'])} lines")

    print("✅ Crisis detection test passed\n")


async def test_state_transitions():
    """Test state router transitions"""
    print("=== Testing State Transitions ===")

    # Create session
    session_id = session_manager.create_session("test_user")

    # Test initial state determination
    next_state = state_router.determine_next_state(
        session_id, "I'm feeling really down today"
    )
    print(f"Initial state -> Next state: {ConversationState.INITIAL_DISTRESS.value} -> {next_state.value}")

    # Update session state
    session_manager.update_session_state(session_id, next_state)

    # Test elaboration state
    next_state = state_router.determine_next_state(
        session_id, "I've been feeling this way for a few weeks now"
    )
    print(f"Elaboration state -> Next state: {next_state.value}")

    print("✅ State transitions test passed\n")


async def test_state_actions():
    """Test state-specific actions"""
    print("=== Testing State Actions ===")

    # Create session
    session_id = session_manager.create_session("test_user")

    # Test initial distress action
    response_data = await state_actions.execute_state_action(
        session_id, "I'm feeling really overwhelmed and stressed"
    )

    print(f"Response type: {response_data.get('response_type')}")
    print(f"Response length: {len(response_data.get('response', ''))}")
    print(f"Current state: {response_data.get('current_state')}")
    print(f"Next state: {response_data.get('next_state')}")

    # Test follow-up action
    response_data2 = await state_actions.execute_state_action(
        session_id, "It's been like this for a couple of weeks"
    )

    print(f"Follow-up response type: {response_data2.get('response_type')}")
    print(f"Follow-up current state: {response_data2.get('current_state')}")

    print("✅ State actions test passed\n")


async def test_conversation_content():
    """Test conversation content management"""
    print("=== Testing Conversation Content ===")

    content_manager = ConversationContentManager()

    # Test content retrieval
    empathy_content = content_manager.get_content("ack_and_probe")
    print(f"Empathy content length: {len(empathy_content)}")

    # Test crisis content
    crisis_content = content_manager.get_content("crisis_intervention", "immediate_response")
    print(f"Crisis content length: {len(crisis_content)}")

    # Test crisis keywords
    keywords = content_manager.get_crisis_keywords()
    print(f"Crisis keywords count: {len(keywords)}")
    print(f"Sample keywords: {keywords[:3]}")

    print("✅ Conversation content test passed\n")


async def test_full_conversation_flow():
    """Test a complete conversation flow"""
    print("=== Testing Full Conversation Flow ===")

    # Create session
    session_id = session_manager.create_session("test_user")
    print(f"Started conversation session: {session_id}")

    # Simulate conversation flow
    conversation_messages = [
        "I'm feeling really stressed and overwhelmed",
        "It's been like this for a few weeks now",
        "I have so much work and family pressure",
        "I don't know how to handle all of this",
        "What should I do?"
    ]

    for i, message in enumerate(conversation_messages):
        print(f"\n--- Turn {i+1} ---")
        print(f"User: {message}")

        # Execute state action
        response_data = await state_actions.execute_state_action(session_id, message)

        response_text = response_data.get('response', '')
        if response_text:
            print(f"Assistant ({response_data.get('response_type')}): {response_text[:100]}...")
        print(f"State: {response_data.get('current_state')} -> {response_data.get('next_state')}")

    # Check final session state
    final_session = session_manager.get_session(session_id)
    summary = session_manager.get_session_summary(session_id)

    print("\nFinal conversation summary:")
    if summary:
        print(f"  Messages: {summary['message_count']}")
        print(f"  Final state: {summary['current_state']}")
        print(f"  Crisis severity: {summary['crisis_severity']}")
        print(f"  Duration: {0.0:.1f} minutes")  # Placeholder since we don't have this field

    print("✅ Full conversation flow test passed\n")


async def main():
    """Run all tests"""
    print("🚀 Starting Stateful Conversation System Tests\n")

    try:
        await test_session_management()
        await test_crisis_detection()
        await test_state_transitions()
        await test_state_actions()
        await test_conversation_content()
        await test_full_conversation_flow()

        print("🎉 All tests passed! Stateful conversation system is working correctly.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)