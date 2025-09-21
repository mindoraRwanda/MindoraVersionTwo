from fastapi import APIRouter, Depends, HTTPException, Query
import bleach
import time
import json
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from backend.app.auth.utils import get_current_user
from backend.app.db.database import SessionLocal
from backend.app.db.models import Conversation, Message, User, EmotionLog
from backend.app.auth.schemas import MessageCreate, MessageOut
from backend.app.services.emotion_classifier import classify_emotion
from backend.app.services.query_validator_langgraph import LangGraphQueryValidator
from backend.app.services.langgraph_state import QueryType

# Import the global LLM service instance (will be initialized on startup)
def get_llm_service():
    """Get the global LLM service instance."""
    from backend.app.main import llm_service
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail="LLM service not initialized. Please try again later."
        )
    return llm_service

def get_query_validator():
    """Get the global query validator instance."""
    from backend.app.main import query_validator
    if not query_validator:
        raise HTTPException(
            status_code=503,
            detail="Query validator not initialized. Please try again later."
        )
    return query_validator

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
    query_validator: LangGraphQueryValidator = Depends(get_query_validator)
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

    # Step 1: Execute the LangGraph workflow
    validation_result = None
    try:
        validation_result = await query_validator.execute_workflow(
            clean_content,
            conversation_history=conversation_history
        )

        # Step 2: Check if conversation should proceed
        should_proceed = validation_result.get("should_proceed_to_conversation", False)
        query_type = validation_result.get("query_type", "unclear")
        is_crisis = validation_result.get("is_crisis", False)
        routing_decision = validation_result.get("routing_decision", "standard_processing")

        print(f"Query validation result: {json.dumps(validation_result, indent=2)}")

        # Step 3: Handle different query types based on routing decision
        if routing_decision == "crisis_intervention":
            # Crisis situation - route to crisis intervention
            bot_reply = await handle_crisis_query(clean_content, validation_result)
        elif routing_decision == "mental_health_support":
            # Mental health query - process with main LLM with emotion context
            llm_start = time.time()
            emotion_data = validation_result.get("emotion_detection", {})
            bot_reply = await llm_service.generate_response(clean_content, conversation_history, str(user.id), skip_analysis=False, emotion_data=emotion_data)
            llm_time = time.time() - llm_start
            print(f"⏱️  LLM generation: {llm_time:.3f}s ({len(bot_reply)} chars)")
        elif routing_decision == "random_question_filtered":
            # Random question - return filtered response
            bot_reply = await handle_random_question(clean_content, validation_result)
        elif routing_decision == "clarification_needed":
            # Unclear query - ask for clarification
            bot_reply = await handle_unclear_query(clean_content, validation_result)
        elif routing_decision == "standard_processing":
            # Default processing - use main LLM with emotion context
            llm_start = time.time()
            emotion_data = validation_result.get("emotion_detection", {})
            bot_reply = await llm_service.generate_response(clean_content, conversation_history, str(user.id), skip_analysis=False, emotion_data=emotion_data)
            llm_time = time.time() - llm_start
            print(f"⏱️  LLM generation: {llm_time:.3f}s ({len(bot_reply)} chars)")
        else:
            # Fallback for unknown routing decisions
            bot_reply = validation_result.get("final_response", "I'm here to help with mental health and emotional support. Could you please clarify what you'd like help with?")

    except Exception as e:
        # Fallback to original behavior if validation fails
        print(f"Query validation failed: {e}")
        llm_start = time.time()
        # No emotion data available in fallback case
        bot_reply = await llm_service.generate_response(clean_content, conversation_history, str(user.id))
        llm_time = time.time() - llm_start
        print(f"⏱️  Fallback LLM generation: {llm_time:.3f}s ({len(bot_reply)} chars)")

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


async def handle_crisis_query(query: str, validation_result: Dict[str, Any]) -> str:
    """
    Handle crisis queries with appropriate intervention.

    Args:
        query: The crisis query
        validation_result: Validation results from LangGraph

    Returns:
        Crisis intervention response
    """
    crisis_severity = validation_result.get("crisis_severity", "medium")

    if crisis_severity == "critical":
        return (
            "🚨 I detect this may be a crisis situation. Your safety is the highest priority.\n\n"
            "Please reach out for immediate professional help:\n"
            "• Emergency Services: 112\n"
            "• Mental Health Helpline: 114 (24/7, free, confidential)\n"
            "• Ndera Neuropsychiatric Hospital: +250 781 447 928\n\n"
            "You don't have to face this alone. Professional crisis support is essential right now. "
            "Please contact emergency services immediately."
        )
    else:
        return (
            "I'm concerned about what you're sharing. While this may not be an immediate crisis, "
            "it's important to talk to a professional.\n\n"
            "Available resources:\n"
            "• Mental Health Helpline: 114\n"
            "• Emergency Services: 112\n"
            "• Local health centers in your area\n\n"
            "Would you like me to help you connect with these services?"
        )


async def handle_random_question(query: str, validation_result: Dict[str, Any]) -> str:
    """
    Handle random questions with filtered responses.

    Args:
        query: The random question
        validation_result: Validation results from LangGraph

    Returns:
        Filtered response for random questions
    """
    suggestions = validation_result.get("suggestions", [])

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
            "I notice you're asking about a technical or programming topic. "
            "While I'm here to support your mental health and emotional well-being, "
            "I'm not designed to provide technical assistance or programming help.\n\n"
            f"Your question: '{query}'\n\n"
            "For technical questions, I recommend:\n"
            "• Consulting official documentation\n"
            "• Using search engines like Google\n"
            "• Asking in programming communities (Stack Overflow, Reddit)\n"
            "• Using AI assistants designed for coding\n\n"
            "However, if this technical issue is causing you stress or affecting your mental health, "
            "I'm here to help you cope with those feelings. Would you like to talk about how this is impacting you emotionally?"
        )
    else:
        return (
            "I understand you have a question, but I'm primarily designed to support mental health and emotional well-being. "
            "While I can try to help with general questions, my expertise is in providing mental health support.\n\n"
            f"Your question: '{query}'\n\n"
            "If this is related to your mental health or emotional well-being, please let me know how I can support you. "
            "For other topics, you might want to consult a general AI assistant or search engine.\n\n"
            "Is there anything related to your mental health or well-being you'd like to discuss?"
        )


async def handle_unclear_query(query: str, validation_result: Dict[str, Any]) -> str:
    """
    Handle unclear queries by asking for clarification.

    Args:
        query: The unclear query
        validation_result: Validation results from LangGraph

    Returns:
        Clarification request response
    """
    return (
        "I'm not sure I understand your query. Could you please clarify what you'd like help with?\n\n"
        "I'm here to support your mental health and emotional well-being. You can ask me about:\n"
        "• How you're feeling emotionally\n"
        "• Stress or anxiety you're experiencing\n"
        "• Depression or mood concerns\n"
        "• Relationship difficulties\n"
        "• Work or school-related stress\n"
        "• General mental health questions\n\n"
        "What would be most helpful for you right now?"
    )