from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
import bleach
import time
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from ..auth.utils import get_current_user
from ..db.database import SessionLocal
from ..db.models import Conversation, Message, User, EmotionLog
from ..auth.schemas import MessageCreate, MessageOut, UserOut
# Removed: unified_conversation_workflow (consolidated into stateful_pipeline)
from ..services.stateful_pipeline import StatefulMentalHealthPipeline

# Import the new stateful conversation system
from ..services.session_state_manager import session_manager
from ..dependencies import get_stateful_pipeline

router = APIRouter(prefix="/auth", tags=["Messages"])

# Dependency: get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Message Handling Endpoints ---

@router.post("/messages")
async def send_message(
    message: MessageCreate,
    background: BackgroundTasks,
    user: UserOut = Depends(get_current_user),
    db: Session = Depends(get_db),
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
    stateful_pipeline = get_stateful_pipeline(db=db, background=background)
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

    # Save user message to database BEFORE pipeline processing to ensure message_id is available for crisis logging
    user_msg = Message(
        conversation_id=convo.id,
        sender="user",
        content=clean_content
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    print(f"💾 User message saved with ID: {user_msg.uuid}")

    # Use the stateful mental health pipeline for end-to-end processing
    workflow_start = time.time()
    pipeline_result = {}  # Initialize to avoid unbound variable

    try:
        # Execute the stateful pipeline - handles validation, crisis detection, emotion detection, and response generation
        pipeline_result = await stateful_pipeline.process_query(
            query=clean_content,
            user_id=str(user.id),
            conversation_id=str(convo.id),
            message_id=str(user_msg.id),
            conversation_history=conversation_history,
            user_gender=str(user.gender),  # Pass user gender for cultural context
            db=db,
            background=background
        )

        # Extract structured output first - this is the primary source of content
        structured_output = pipeline_result.get("assistant_structured_output", {})
        if structured_output and isinstance(structured_output, dict):
            # Use the message from structured output as primary content
            bot_reply = structured_output.get("message", pipeline_result.get("response", "I'm here to support you."))
        else:
            bot_reply = pipeline_result.get("response", "I'm here to support you.")
        
        # Ensure bot_reply is not empty
        if not bot_reply or not str(bot_reply).strip():
            print("⚠️  Warning: Pipeline returned empty response, using fallback")
            bot_reply = "I'm here to support you. How can I help you today?"
        
        # Extract all metadata
        diagnostic_slots = pipeline_result.get("diagnostic_slots", {})
        processing_metadata = pipeline_result.get("processing_metadata", [])
        response_confidence = pipeline_result.get("response_confidence", 0.0)
        response_reason = pipeline_result.get("response_reason", "")
        query_validation = pipeline_result.get("query_validation")
        crisis_assessment = pipeline_result.get("crisis_assessment")
        emotion_detection = pipeline_result.get("emotion_detection")
        query_evaluation = pipeline_result.get("query_evaluation")
        cultural_context_applied = pipeline_result.get("cultural_context_applied", [])
        processing_time = pipeline_result.get("processing_time", 0.0)
        llm_calls_made = pipeline_result.get("llm_calls_made", 0)
        errors = pipeline_result.get("errors", [])

        workflow_time = time.time() - workflow_start
        print(f"⏱️  Stateful pipeline processing: {workflow_time:.3f}s ({len(str(bot_reply))} chars)")
        print(f"🤖 Response confidence: {response_confidence:.2f}")
        print(f"📊 Processing steps: {len(processing_metadata)}")
        print(f"📋 Structured output available: {bool(structured_output)}")
        if structured_output:
            print(f"📝 Structured message: {structured_output.get('message', 'N/A')[:100]}...")

    except Exception as e:
        import traceback
        print(f"Stateful pipeline failed: {e}")
        traceback.print_exc()
        # Final fallback to basic response
        bot_reply = "I'm here to support you. How can I help you today?"
        workflow_time = time.time() - workflow_start
        print(f"⏱️  Fallback processing: {workflow_time:.3f}s")
        # Initialize empty metadata for fallback
        structured_output = {}
        diagnostic_slots = {}
        processing_metadata = []
        response_confidence = 0.0
        response_reason = f"Pipeline error: {str(e)}"
        query_validation = None
        crisis_assessment = None
        emotion_detection = None
        query_evaluation = None
        cultural_context_applied = []
        processing_time = workflow_time
        llm_calls_made = 0
        errors = [str(e)]

    # Get emotion data from stateful pipeline result
    detected_emotion = "neutral"
    if emotion_detection and hasattr(emotion_detection, 'selected_emotion'):
        detected_emotion = emotion_detection.selected_emotion
    elif isinstance(emotion_detection, dict):
        detected_emotion = emotion_detection.get('selected_emotion', 'neutral')
    
    # Get query evaluation data for routing decision
    routing_decision = "GIVE_EMPATHY"
    if query_evaluation and hasattr(query_evaluation, 'evaluation_type'):
        routing_decision = query_evaluation.evaluation_type.value
    elif isinstance(query_evaluation, dict):
        routing_decision = query_evaluation.get('evaluation_type', {}).get('value', 'GIVE_EMPATHY')

    # Batch database operations - create all objects first
    db_prep_start = time.time()

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
    db.add_all([bot_msg, emotion_log])
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

    # Helper function to serialize complex objects
    def serialize_obj(obj):
        """Serialize objects that might have attributes or be dicts."""
        if obj is None:
            return None
        if hasattr(obj, 'dict'):  # Pydantic models
            return obj.dict()
        elif hasattr(obj, '__dict__'):  # Regular objects
            return {k: serialize_obj(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        elif isinstance(obj, dict):
            return {k: serialize_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_obj(item) for item in obj]
        elif hasattr(obj, 'value'):  # Enum-like
            return obj.value
        else:
            return obj
    
    # Use structured output: combine message + question_next for the response content
    response_content = bot_reply
    if structured_output and isinstance(structured_output, dict):
        structured_message = structured_output.get("message", "").strip()
        question_next = structured_output.get("question_next", "").strip()
        
        # Combine message and question_next
        if structured_message:
            if question_next:
                # Combine with a space
                response_content = f"{structured_message} {question_next}"
            else:
                response_content = structured_message
    
    # Ensure we have content
    if not response_content or not str(response_content).strip():
        response_content = "I'm here to support you. How can I help you today?"
    
    # Return response in format expected by frontend, with structured output and metadata
    return {
        "response": {
            "content": response_content,
            "timestamp": bot_msg.timestamp.isoformat() if hasattr(bot_msg.timestamp, 'isoformat') else str(bot_msg.timestamp)
        },
        "emotion": detected_emotion if detected_emotion != "neutral" else None,
        # Structured output from core chat (AssistantTurnState)
        "structured_output": serialize_obj(structured_output) if structured_output else None,
        # Comprehensive metadata
        "metadata": {
            "response_confidence": response_confidence,
            "response_reason": response_reason,
            "diagnostic_slots": diagnostic_slots,
            "processing_metadata": [
                {
                    "step": meta.get("step", meta.get("step_name", "")) if isinstance(meta, dict) else getattr(meta, "step_name", ""),
                    "confidence": meta.get("confidence", meta.get("confidence_score", 0.0)) if isinstance(meta, dict) else getattr(meta, "confidence_score", 0.0),
                    "reasoning": meta.get("reasoning", "") if isinstance(meta, dict) else getattr(meta, "reasoning", ""),
                    "keywords": meta.get("keywords", []) if isinstance(meta, dict) else getattr(meta, "keywords", []),
                    "processing_time": meta.get("processing_time", 0.0) if isinstance(meta, dict) else getattr(meta, "processing_time", 0.0),
                }
                for meta in processing_metadata
            ],
            "query_validation": serialize_obj(query_validation) if query_validation else None,
            "crisis_assessment": serialize_obj(crisis_assessment) if crisis_assessment else None,
            "emotion_detection": serialize_obj(emotion_detection) if emotion_detection else None,
            "query_evaluation": serialize_obj(query_evaluation) if query_evaluation else None,
            "cultural_context_applied": cultural_context_applied,
            "routing_decision": routing_decision,
            "processing_time": processing_time,
            "llm_calls_made": llm_calls_made,
            "errors": errors if errors else [],
        }
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

