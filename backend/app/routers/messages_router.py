from fastapi import APIRouter, Depends, HTTPException, Query
import bleach
import time
import json
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from ..auth.utils import get_current_user
from ..settings.settings import settings
from ..db.database import SessionLocal
from ..db.models import Conversation, Message, User, EmotionLog
from ..auth.schemas import MessageCreate, MessageOut
# Removed: unified_conversation_workflow (consolidated into stateful_pipeline)
from backend.app.services.stateful_pipeline import StatefulMentalHealthPipeline

# Import the new stateful conversation system
from backend.app.services.session_state_manager import session_manager

# Import service container for dependency injection
def get_llm_service():
    """Get LLM service from service container."""
    from ..services.service_container import get_service
    return get_service("llm_service")

# Removed: get_unified_workflow (consolidated into stateful_pipeline)

def get_stateful_pipeline():
    """Get stateful mental health pipeline from service container."""
    from ..services.service_container import get_service
    try:
        return get_service("stateful_pipeline")
    except Exception:
        # Fallback to creating a new instance
        from ..services.stateful_pipeline import initialize_stateful_pipeline
        llm_service = get_service("llm_service")
        return initialize_stateful_pipeline(llm_provider=llm_service.llm_provider if llm_service else None)

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
    stateful_pipeline: StatefulMentalHealthPipeline = Depends(get_stateful_pipeline)
):
    """
    Enhanced message endpoint with stateful LangGraph mental health pipeline.

    This endpoint now uses a stateful LangGraph pipeline that provides:
    1. Comprehensive query validation with confidence scoring
    2. Crisis detection with severity classification
    3. Emotion detection with youth-specific patterns
    4. Specialized response nodes for different strategies
    5. Full explainability and transparency
    6. Cultural context integration throughout

    The pipeline provides complete explainability for all processing decisions.
    """
    pipeline_start = time.time()
    print(f"\n🚀 Starting enhanced message pipeline for user {user.id}")

    # Single query to verify conversation ownership
    db_start = time.time()
    convo = db.query(Conversation).filter_by(
        uuid=message.conversation_id,
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
        .filter_by(conversation_id=convo.id)\
        .order_by(Message.timestamp.desc())\
        .limit(15)\
        .all()

    recent_history.reverse()
    conversation_history = [
        {"role": msg.sender, "text": msg.content} for msg in recent_history
    ]
    history_time = time.time() - history_start
    print(f"⏱️  DB history load: {history_time:.3f}s ({len(recent_history)} messages)")

    # Use the stateful mental health pipeline for end-to-end processing
    workflow_start = time.time()
    pipeline_result = {}  # Initialize to avoid unbound variable

    try:
        # Execute the stateful pipeline - handles validation, crisis detection, emotion detection, and response generation
        pipeline_result = await stateful_pipeline.process_query(
            query=clean_content,
            user_id=str(user.id),
            conversation_history=conversation_history,
            user_gender=user.gender  # Pass user gender for cultural context
        )

        bot_reply = pipeline_result.get("response", "I'm here to support you.")
        response_confidence = pipeline_result.get("response_confidence", 0.0)
        processing_metadata = pipeline_result.get("processing_metadata", [])

        workflow_time = time.time() - workflow_start
        print(f"⏱️  Stateful pipeline processing: {workflow_time:.3f}s ({len(bot_reply)} chars)")
        print(f"🤖 Response confidence: {response_confidence:.2f}")
        print(f"📊 Processing steps: {len(processing_metadata)}")

    except Exception as e:
        print(f"Stateful pipeline failed: {e}")
        # Final fallback to basic response
        bot_reply = "I'm here to support you. How can I help you today?"
        workflow_time = time.time() - workflow_start
        print(f"⏱️  Fallback processing: {workflow_time:.3f}s")

    # Get emotion data from stateful pipeline result
    emotion_detection = pipeline_result.get("emotion_detection") if 'pipeline_result' in locals() else None
    detected_emotion = emotion_detection.selected_emotion if emotion_detection and hasattr(emotion_detection, 'selected_emotion') else "neutral"
    
    # Get query evaluation data for routing decision
    query_evaluation = pipeline_result.get("query_evaluation") if 'pipeline_result' in locals() else None
    routing_decision = query_evaluation.evaluation_type.value if query_evaluation and hasattr(query_evaluation, 'evaluation_type') else "GIVE_EMPATHY"

    # Batch database operations - create all objects first
    db_prep_start = time.time()
    user_msg = Message(
        conversation_id=convo.id,
        sender="user",
        content=clean_content
    )

    bot_msg = Message(
        conversation_id=convo.id,
        sender="bot",
        content=bleach.clean(bot_reply)
    )

    emotion_log = EmotionLog(
        user_id=user.id,
        conversation_id=convo.id,
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

    # The unified workflow already handles state management, so we just need to add to history
    session_manager.add_message_to_history(
        str(convo.id),
        "assistant",
        bot_reply,
        {
            "routing_decision": routing_decision,
            "emotion_detected": detected_emotion,
            "response_type": "stateful_pipeline",
            "pipeline_result": pipeline_result
        }
    )
    session_update_time = time.time() - session_update_start
    print(f"⏱️  Session state update: {session_update_time:.3f}s")

    total_time = time.time() - pipeline_start
    print(f"🏁 Total pipeline time: {total_time:.3f}s")
    print(f"📊 Breakdown: DB({db_lookup_time + history_time + db_save_time:.3f}s) | LLM/Validation({total_time - (db_lookup_time + history_time + db_save_time):.3f}s)")

    # Debug: Log chunking info
    should_chunk = pipeline_result.get("should_chunk", False)
    response_chunks = pipeline_result.get("response_chunks", [])
    print(f"📦 API Response - should_chunk: {should_chunk}, chunks: {len(response_chunks)}")
    if response_chunks:
        print(f"📦 First chunk preview: {response_chunks[0] if response_chunks else 'None'}")

    return {
        "id": bot_msg.uuid,
        "sender": bot_msg.sender,
        "content": bot_msg.content,
        "timestamp": bot_msg.timestamp,
        "should_chunk": pipeline_result.get("should_chunk", False),
        "response_chunks": pipeline_result.get("response_chunks", [])
    }


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

    # Return messages with uuid instead of id
    return [
        {
            "id": msg.uuid,
            "sender": msg.sender,
            "content": msg.content,
            "timestamp": msg.timestamp
        } for msg in reversed(messages)
    ]

    return list(reversed(messages))  # return oldest → newest

