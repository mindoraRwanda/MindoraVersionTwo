from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
import bleach
import time
import json
from collections import defaultdict, deque
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from ..auth.utils import get_current_user
from ..settings.settings import settings
from ..db.database import SessionLocal
from ..db.models import Conversation, Message, User, EmotionLog
from ..auth.schemas import MessageCreate, MessageOut, UserOut
from ..services.stateful_pipeline import StatefulMentalHealthPipeline

from ..services.session_state_manager import session_manager
from ..dependencies import get_stateful_pipeline

router = APIRouter(prefix="/auth", tags=["Messages"])

# ── Per-user rate limiter (sliding-window, in-memory) ──────────
_RATE_WINDOW_SECONDS = 60
_RATE_MAX_REQUESTS   = 20            # 20 messages per minute per user
_rate_store: Dict[str, deque] = defaultdict(deque)

def _check_rate_limit(user_id: str) -> None:
    """Raise 429 if user has exceeded the per-minute message cap."""
    now = time.time()
    q = _rate_store[user_id]
    # Drop timestamps outside the window
    while q and q[0] < now - _RATE_WINDOW_SECONDS:
        q.popleft()
    if len(q) >= _RATE_MAX_REQUESTS:
        retry_in = int(_RATE_WINDOW_SECONDS - (now - q[0])) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Too many messages. Please wait {retry_in} seconds before sending again.",
            headers={"Retry-After": str(retry_in)},
        )
    q.append(now)
# ──────────────────────────────────────────────────────────────

# Dependency: get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def _generate_conversation_title(
    llm_provider,
    user_message: str,
    bot_reply: str,
    conversation_id: int,
) -> None:
    """Generate a 4-6 word title after the first exchange and persist it in meta."""
    from ..db.database import SessionLocal as _SessionLocal
    from langchain_core.messages import HumanMessage, SystemMessage

    db = _SessionLocal()
    try:
        msgs = [
            SystemMessage(content=(
                "Generate a concise 4-6 word title for a therapy conversation based on "
                "the first message and reply. Return ONLY the title — no quotes, no period. "
                "Examples: 'Job loss and self-worth', 'Anxiety about the future', "
                "'Grief after losing a parent', 'Struggling with loneliness'"
            )),
            HumanMessage(content=(
                f"User: {user_message[:300]}\nBot: {bot_reply[:300]}"
            )),
        ]
        raw = await llm_provider.agenerate(msgs)
        title = str(raw).strip().strip("\"'").strip()[:80]

        convo = db.query(Conversation).filter_by(id=conversation_id).first()
        if convo:
            meta = dict(convo.meta or {})
            meta["title"] = title
            convo.meta = meta
            db.commit()
            print(f"✅ Conversation title generated: '{title}'")
    except Exception as exc:
        print(f"⚠️  Title generation failed: {exc}")
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
    _check_rate_limit(str(user.id))
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
        .limit(20)\
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

    # Generate a title after the very first exchange (no prior history)
    if len(conversation_history) == 0 and stateful_pipeline.llm_provider:
        background.add_task(
            _generate_conversation_title,
            stateful_pipeline.llm_provider,
            clean_content,
            bot_reply,
            convo.id,
        )

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

    return {
        "id": bot_msg.uuid,
        "sender": bot_msg.sender.value,
        "content": bot_msg.content,
        "timestamp": bot_msg.timestamp
    }


@router.post("/messages/stream")
async def stream_message(
    message: MessageCreate,
    background: BackgroundTasks,
    user: UserOut = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    SSE endpoint that streams the bot response token-by-token.

    Response format (text/event-stream):
      data: {"token": "Hello"}\n\n
      data: {"token": " there"}\n\n
      data: {"done": true, "id": "...", "timestamp": "..."}\n\n
    """
    _check_rate_limit(str(user.id))
    stateful_pipeline = get_stateful_pipeline(db=db, background=background)

    convo = db.query(Conversation).filter_by(
        uuid=message.conversation_id, user_id=user.id
    ).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    clean_content = bleach.clean(message.content.strip())
    if not clean_content:
        raise HTTPException(status_code=400, detail="Message content is empty or invalid")

    # Load history while the session is still alive
    recent_history = (
        db.query(Message)
        .filter_by(conversation_id=convo.id)
        .order_by(Message.timestamp.desc())
        .limit(20)
        .all()
    )
    recent_history.reverse()
    conversation_history = [{"role": str(m.sender.value if hasattr(m.sender, 'value') else m.sender), "text": m.content} for m in recent_history]

    # ── Extract plain Python values BEFORE any commit ───────────────
    # FastAPI closes the get_db() session when StreamingResponse is returned,
    # so we must not rely on ORM objects inside the async generator.
    convo_pk   = convo.id          # UUID primary key
    user_pk    = user.id
    user_gen   = str(user.gender) if user.gender else None
    is_first_exchange = len(conversation_history) == 0
    # ────────────────────────────────────────────────────────────────

    # Save user message using the still-open session
    user_msg = Message(conversation_id=convo_pk, sender="user", content=clean_content)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    user_msg_pk = user_msg.id      # plain UUID, safe to close session after this

    async def event_generator():
        # Open a FRESH session — the outer db session is closed by now.
        from ..db.database import SessionLocal as _SL
        db2 = _SL()
        full_response = ""
        try:
            async for token in stateful_pipeline.process_query_stream(
                query=clean_content,
                user_id=str(user_pk),
                conversation_id=str(convo_pk),
                message_id=str(user_msg_pk),
                conversation_history=conversation_history,
                user_gender=user_gen,
                db=db2,
                background=background,
            ):
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            print(f"Stream error: {exc}")
            if not full_response:
                full_response = "I'm here to support you. How can I help you today?"
                yield f"data: {json.dumps({'token': full_response})}\n\n"

        # Persist bot message with the fresh session
        safe_reply = bleach.clean(full_response) if full_response else "I'm here to support you."
        try:
            bot_msg = Message(conversation_id=convo_pk, sender="bot", content=safe_reply)
            db2.add(bot_msg)
            # Update last_activity_at by re-fetching convo from the fresh session
            convo2 = db2.get(Conversation, convo_pk)
            if convo2:
                convo2.last_activity_at = bot_msg.timestamp
            db2.commit()
            db2.refresh(bot_msg)

            if is_first_exchange and stateful_pipeline.llm_provider:
                background.add_task(
                    _generate_conversation_title,
                    stateful_pipeline.llm_provider,
                    clean_content,
                    safe_reply,
                    convo_pk,
                )

            yield f"data: {json.dumps({'done': True, 'id': str(bot_msg.uuid), 'timestamp': bot_msg.timestamp.isoformat()})}\n\n"
        except Exception as save_exc:
            print(f"Stream DB save error: {save_exc}")
            yield f"data: {json.dumps({'done': True, 'id': None, 'timestamp': None})}\n\n"
        finally:
            db2.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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

