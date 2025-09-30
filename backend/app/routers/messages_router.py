from fastapi import APIRouter, Depends, HTTPException, Query
import bleach
import time
import json
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from backend.app.auth.utils import get_current_user
from backend.app.db.database import SessionLocal
from backend.app.db.models import Conversation, Message, User, EmotionLog
from backend.app.auth.schemas import MessageCreate, MessageOut
from backend.app.services.emotion_classifier import classify_emotion
from backend.app.services.query_validator_langgraph import LangGraphQueryValidator
from backend.app.services.query_validator import QueryValidatorService
from typing import Union
from backend.app.services.langgraph_state import QueryType

# Import the new stateful conversation system
from backend.app.services.session_state_manager import session_manager
from backend.app.services.langgraph_state_router import llm_enhanced_router

# Import service container for dependency injection
def get_llm_service():
    """Get LLM service from service container."""
    from backend.app.services.service_container import get_service
    return get_service("llm_service")

def get_query_validator():
    """Get query validator from service container."""
    from backend.app.services.service_container import get_service
    return get_service("query_validator")

router = APIRouter(prefix="/auth", tags=["Messages"])

# Dependency: get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Message Handling Endpoints ---

@router.post("/messages", response_model=MessageOut)
async def send_message(
    message: MessageCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    llm_service = Depends(get_llm_service),
    query_validator: Union[LangGraphQueryValidator, QueryValidatorService] = Depends(get_query_validator)
):
    """
    Enhanced message endpoint with LangGraph query validation.

    This endpoint now uses LangGraph query validation to:
    1. Validate if the query is mental health related
    2. Filter out unrelated questions
    3. Route appropriate queries to the main LLM service
    4. Provide contextual responses for off-topic queries
    """
    pipeline_start = time.time()
    print(f"\n🚀 Starting enhanced message pipeline for user {user.id}")

    # Single query to verify conversation ownership
    db_start = time.time()
    convo = db.query(Conversation).filter_by(
        id=message.conversation_id,
        user_id=user.id
    ).first()
    db_lookup_time = time.time() - db_start
    print(f"⏱️  DB conversation lookup: {db_lookup_time:.3f}s")

    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Content cleaning
    clean_start = time.time()
    clean_content = bleach.clean(message.content.strip())
    clean_time = time.time() - clean_start
    print(f"⏱️  Content cleaning: {clean_time:.3f}s")

    if not clean_content:
        raise HTTPException(status_code=400, detail="Message content is empty or invalid")

    # Load conversation history for context
    history_start = time.time()
    recent_history = db.query(Message)\
        .filter_by(conversation_id=message.conversation_id)\
        .order_by(Message.timestamp.desc())\
        .limit(15)\
        .all()

    recent_history.reverse()
    conversation_history = [
        {"role": msg.sender, "text": msg.content} for msg in recent_history
    ]
    history_time = time.time() - history_start
    print(f"⏱️  DB history load: {history_time:.3f}s ({len(recent_history)} messages)")

    # Initialize or update session state for stateful conversation
    session_start = time.time()
    session_id = str(message.conversation_id)

    # Ensure session exists in state manager
    session = session_manager.get_session(session_id)
    if not session:
        session_manager.create_session(str(user.id))
        print(f"🆕 Created new conversation session: {session_id}")
    else:
        print(f"📋 Using existing conversation session: {session_id}")

    session_time = time.time() - session_start
    print(f"⏱️  Session management: {session_time:.3f}s")

    # Step 1: Execute the LangGraph workflow
    validation_result = None
    routing_decision = "standard_processing"  # Default value
    try:
        # Check if this is a LangGraph validator or simple validator
        if hasattr(query_validator, 'execute_workflow'):
            # LangGraph validator
            validation_result = await query_validator.execute_workflow(
                clean_content,
                conversation_history=conversation_history
            )
        else:
            # Simple validator - use basic validation method (not async)
            basic_result = query_validator.validate_query(clean_content)
            # Convert basic result to expected format
            validation_result = {
                "query_type": basic_result.query_type.value,
                "confidence": basic_result.confidence,
                "keywords_found": basic_result.keywords_found,
                "reasoning": basic_result.reasoning,
                "should_proceed_to_conversation": basic_result.query_type.value == "mental_support",
                "is_crisis": False,
                "routing_decision": "mental_health_support" if basic_result.query_type.value == "mental_support" else "random_question_filtered"
            }

        # Step 2: Check if conversation should proceed
        should_proceed = validation_result.get("should_proceed_to_conversation", False)
        query_type = validation_result.get("query_type", "unclear")
        is_crisis = validation_result.get("is_crisis", False)
        routing_decision = validation_result.get("routing_decision", "standard_processing")

        print(f"Query validation result: {json.dumps(validation_result, indent=2)}")

        # Step 3: Handle different query types with enhanced stateful conversation system
        if routing_decision == "crisis_intervention":
            # Crisis situation - route to enhanced stateful crisis intervention
            bot_reply = await handle_crisis_query(clean_content, validation_result, str(message.conversation_id))
        elif routing_decision == "mental_health_support":
            # Mental health query - use LLM-enhanced stateful conversation system
            stateful_start = time.time()
            emotion_data = validation_result.get("emotion_detection", {})
            cultural_context = {"region": "rwanda", "user_id": str(user.id)}

            # Use conversation_id as session_id for stateful conversation
            session_id = str(message.conversation_id)

            # Get stateful conversation response
            stateful_response = await llm_enhanced_router.route_conversation(
                session_id, clean_content, emotion_data, cultural_context
            )

            bot_reply = stateful_response.get("response", "I'm here to support you.")
            stateful_time = time.time() - stateful_start
            print(f"⏱️  Stateful conversation: {stateful_time:.3f}s ({len(bot_reply)} chars)")
            print(f"🤖 State: {stateful_response.get('next_state')} | Confidence: {stateful_response.get('confidence', 0):.2f}")

        elif routing_decision == "random_question_filtered":
            # Random question - return enhanced stateful filtered response
            bot_reply = await handle_random_question(clean_content, validation_result, str(message.conversation_id))
        elif routing_decision == "clarification_needed":
            # Unclear query - ask for enhanced stateful clarification
            bot_reply = await handle_unclear_query(clean_content, validation_result, str(message.conversation_id))
        elif routing_decision == "standard_processing":
            # Default processing - use LLM-enhanced stateful conversation as fallback
            stateful_start = time.time()
            emotion_data = validation_result.get("emotion_detection", {})
            cultural_context = {"region": "rwanda", "user_id": str(user.id)}

            session_id = str(message.conversation_id)
            stateful_response = await llm_enhanced_router.route_conversation(
                session_id, clean_content, emotion_data, cultural_context
            )

            bot_reply = stateful_response.get("response", "I'm here to support you.")
            stateful_time = time.time() - stateful_start
            print(f"⏱️  Fallback stateful conversation: {stateful_time:.3f}s ({len(bot_reply)} chars)")

        else:
            # Fallback for unknown routing decisions
            bot_reply = validation_result.get("final_response", "I'm here to help with mental health and emotional support. Could you please clarify what you'd like help with?")

    except Exception as e:
        # Enhanced fallback with stateful conversation system
        print(f"Query validation failed: {e}")
        routing_decision = "fallback_processing"  # Set fallback routing decision
        try:
            # Try stateful conversation as fallback
            stateful_start = time.time()
            session_id = str(message.conversation_id)
            emotion_data = {"fallback": True}
            cultural_context = {"region": "rwanda", "user_id": str(user.id)}

            stateful_response = await llm_enhanced_router.route_conversation(
                session_id, clean_content, emotion_data, cultural_context
            )

            bot_reply = stateful_response.get("response", "I'm here to support you through this difficult time.")
            stateful_time = time.time() - stateful_start
            print(f"⏱️  Fallback stateful conversation: {stateful_time:.3f}s ({len(bot_reply)} chars)")

        except Exception as e2:
            print(f"Stateful fallback also failed: {e2}")
            # Final fallback to original LLM behavior
            llm_start = time.time()
            bot_reply = await llm_service.generate_response(clean_content, conversation_history, str(user.id))
            llm_time = time.time() - llm_start
            print(f"⏱️  Final fallback LLM generation: {llm_time:.3f}s ({len(bot_reply)} chars)")

    # Get emotion data from LangGraph validation result
    emotion_detection = validation_result.get("emotion_detection", {}) if validation_result else {}
    detected_emotion = emotion_detection.get("detected_emotion", "neutral") if isinstance(emotion_detection, dict) else "neutral"

    # Batch database operations - create all objects first
    db_prep_start = time.time()
    user_msg = Message(
        conversation_id=message.conversation_id,
        sender="user",
        content=clean_content
    )

    bot_msg = Message(
        conversation_id=message.conversation_id,
        sender="bot",
        content=bleach.clean(bot_reply)
    )

    emotion_log = EmotionLog(
        user_id=user.id,
        conversation_id=message.conversation_id,
        input_text=clean_content,
        detected_emotion=detected_emotion
    )

    # Single transaction - add all objects and commit once
    db.add_all([user_msg, bot_msg, emotion_log])
    convo.last_activity_at = bot_msg.timestamp
    db.commit()

    # Only refresh the bot message we're returning
    db.refresh(bot_msg)
    db_save_time = time.time() - db_prep_start
    print(f"⏱️  DB save operations: {db_save_time:.3f}s")

    # Update session activity in state manager
    session_update_start = time.time()
    session_manager.add_message_to_history(
        session_id,
        "assistant",
        bot_reply,
        {
            "routing_decision": routing_decision,
            "emotion_detected": detected_emotion,
            "response_type": "stateful_conversation" if routing_decision == "mental_health_support" else routing_decision
        }
    )
    session_update_time = time.time() - session_update_start
    print(f"⏱️  Session state update: {session_update_time:.3f}s")

    total_time = time.time() - pipeline_start
    print(f"🏁 Total pipeline time: {total_time:.3f}s")
    print(f"📊 Breakdown: DB({db_lookup_time + history_time + db_save_time:.3f}s) | LLM/Validation({total_time - (db_lookup_time + history_time + db_save_time):.3f}s)")

    return bot_msg


@router.get("/context", response_model=List[MessageOut])
def get_context_window(
    limit: int = Query(default=10, le=50),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Get context window of recent messages for the user."""
    # Get most recently active conversation (with messages)
    conversation = (
        db.query(Conversation)
        .filter_by(user_id=user.id)
        .order_by(Conversation.started_at.desc())
        .first()
    )

    if not conversation:
        return []

    messages = (
        db.query(Message)
        .filter_by(conversation_id=conversation.id)
        .order_by(Message.timestamp.desc())
        .limit(limit)
        .all()
    )

    return list(reversed(messages))  # return oldest → newest


async def handle_crisis_query(query: str, validation_result: Dict[str, Any], conversation_id: Optional[str] = None) -> str:
    """
    Handle crisis queries with enhanced stateful crisis intervention.

    Args:
        query: The crisis query
        validation_result: Validation results from LangGraph
        conversation_id: Conversation ID for stateful session management

    Returns:
        Enhanced crisis intervention response with stateful conversation context
    """
    crisis_severity = validation_result.get("crisis_severity", "medium")

    # Use stateful conversation system for crisis intervention
    if conversation_id:
        try:
            # Get emotion data for crisis context
            emotion_data = validation_result.get("emotion_detection", {})
            cultural_context = {"region": "rwanda", "crisis": True}

            # Use stateful crisis intervention
            crisis_response = await llm_enhanced_router.route_conversation(
                conversation_id, query, emotion_data, cultural_context
            )

            if crisis_response and crisis_response.get("response"):
                return crisis_response["response"]
        except Exception as e:
            print(f"Stateful crisis intervention failed: {e}")
            # Fall back to basic crisis response

    # Enhanced crisis responses with more detailed information
    if crisis_severity == "critical":
        return (
            "🚨 **CRISIS INTERVENTION** 🚨\n\n"
            "I detect this may be a crisis situation. Your safety is the highest priority.\n\n"
            "**Please reach out for immediate professional help:**\n\n"
            "• **Emergency Services: 112** (Call now)\n"
            "• **Mental Health Helpline: 114** (24/7, free, confidential)\n"
            "• **Ndera Neuropsychiatric Hospital: +250 781 447 928**\n"
            "• **CARITAS Rwanda Crisis Support: +250 788 123 456**\n\n"
            "You don't have to face this alone. Professional crisis support is essential right now. "
            "Please contact emergency services immediately.\n\n"
            "*Your life matters and there are people who care about you.*"
        )
    else:
        return (
            "I'm very concerned about what you're sharing. While this may not be an immediate crisis, "
            "it's important to talk to a professional who can provide the support you need.\n\n"
            "**Available resources:**\n"
            "• **Mental Health Helpline: 114** (24/7 support)\n"
            "• **Emergency Services: 112** (for immediate danger)\n"
            "• **Local health centers** in your area\n"
            "• **Rwanda Mental Health Association** for ongoing support\n\n"
            "Would you like to talk more about what's going on, or would you prefer to contact one of these services right away?\n\n"
            "*Remember: Seeking help is a sign of strength, not weakness.*"
        )


async def handle_random_question(query: str, validation_result: Dict[str, Any], conversation_id: Optional[str] = None) -> str:
    """
    Handle random questions with enhanced stateful filtered responses.

    Args:
        query: The random question
        validation_result: Validation results from LangGraph
        conversation_id: Conversation ID for stateful session management

    Returns:
        Enhanced filtered response with stateful conversation context
    """
    suggestions = validation_result.get("suggestions", [])

    # Use stateful conversation for filtered responses if conversation_id provided
    if conversation_id:
        try:
            # Create context for filtered response
            emotion_data = {"filtered": True, "question_type": "random"}
            cultural_context = {"region": "rwanda", "filtered": True}

            filtered_response = await llm_enhanced_router.route_conversation(
                conversation_id, query, emotion_data, cultural_context
            )

            if filtered_response and filtered_response.get("response"):
                return filtered_response["response"]
        except Exception as e:
            print(f"Stateful filtered response failed: {e}")
            # Fall back to basic filtered response

    # Check if this is a technical/programming question
    technical_keywords = [
        'python', 'javascript', 'java', 'programming', 'coding', 'software',
        'computer', 'install', 'setup', 'configuration', 'bug', 'error',
        'debug', 'code', 'script', 'algorithm', 'function', 'variable',
        'class', 'method', 'api', 'framework', 'library', 'package'
    ]

    is_technical = any(keyword in query.lower() for keyword in technical_keywords)

    if is_technical:
        return (
            "**Technical Question Detected** 💻\n\n"
            "I notice you're asking about a technical or programming topic. "
            "While I'm here to support your mental health and emotional well-being, "
            "I'm not designed to provide technical assistance or programming help.\n\n"
            f"**Your question:** '{query}'\n\n"
            "**For technical questions, I recommend:**\n"
            "• 📚 Consulting official documentation\n"
            "• 🔍 Using search engines like Google\n"
            "• 👥 Asking in programming communities (Stack Overflow, Reddit)\n"
            "• 🤖 Using AI assistants designed for coding\n\n"
            "---\n\n"
            "However, if this technical issue is causing you **stress** or affecting your **mental health**, "
            "I'm here to help you cope with those feelings. \n\n"
            "Would you like to talk about how this is impacting you emotionally? "
            "*I'm here to listen and support you.*"
        )
    else:
        return (
            "**Question Received** ❓\n\n"
            "I understand you have a question, but I'm primarily designed to support **mental health and emotional well-being**. "
            "While I can try to help with general questions, my expertise is in providing mental health support.\n\n"
            f"**Your question:** '{query}'\n\n"
            "**If this is related to your mental health or emotional well-being, please let me know how I can support you.** "
            "For other topics, you might want to consult a general AI assistant or search engine.\n\n"
            "---\n\n"
            "**Is there anything related to your mental health or well-being you'd like to discuss?**\n\n"
            "*I'm here to listen without judgment.*"
        )


async def handle_unclear_query(query: str, validation_result: Dict[str, Any], conversation_id: Optional[str] = None) -> str:
    """
    Handle unclear queries with enhanced stateful clarification requests.

    Args:
        query: The unclear query
        validation_result: Validation results from LangGraph
        conversation_id: Conversation ID for stateful session management

    Returns:
        Enhanced clarification request with stateful conversation context
    """
    # Use stateful conversation for clarification if conversation_id provided
    if conversation_id:
        try:
            # Create context for clarification response
            emotion_data = {"unclear": True, "needs_clarification": True}
            cultural_context = {"region": "rwanda", "clarification": True}

            clarification_response = await llm_enhanced_router.route_conversation(
                conversation_id, query, emotion_data, cultural_context
            )

            if clarification_response and clarification_response.get("response"):
                return clarification_response["response"]
        except Exception as e:
            print(f"Stateful clarification response failed: {e}")
            # Fall back to basic clarification response

    return (
        "**I Want to Help You** 💙\n\n"
        "I'm not sure I understand your query clearly. Could you please clarify what you'd like help with?\n\n"
        "**I'm here to support your mental health and emotional well-being.** You can ask me about:\n\n"
        "• 💭 How you're feeling emotionally\n"
        "• 😰 Stress or anxiety you're experiencing\n"
        "• 😔 Depression or mood concerns\n"
        "• 👥 Relationship difficulties\n"
        "• 💼 Work or school-related stress\n"
        "• 🧠 General mental health questions\n\n"
        "---\n\n"
        "**What would be most helpful for you right now?**\n\n"
        "*I'm here to listen and support you without judgment.*"
    )