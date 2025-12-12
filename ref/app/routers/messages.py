"""Message endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
import time
from app.database import get_db
from app.core.security import get_current_active_user
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message, MessageRole
from app.schemas.message import MessageCreate, MessageResponse, MessageListResponse
from app.services.llm import chat_structured, build_messages
from app.services.safety import classify_safety
from app.services.slots import apply_slot_updates
from app.utils.logging import write_detailed_log, write_conversation_log, save_conversation_snapshot, now_iso
from app.config import get_settings

router = APIRouter(prefix="/messages", tags=["messages"])
settings = get_settings()


@router.get("", response_model=MessageListResponse)
def get_messages(
    conversation_id: str = Query(..., description="Conversation ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get conversation messages."""
    # Verify conversation belongs to user
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.sequence_number).offset(skip).limit(limit).all()
    
    total = db.query(Message).filter(Message.conversation_id == conversation_id).count()
    
    return {
        "messages": messages,
        "total": total,
        "conversation_id": conversation_id
    }


@router.post("", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def send_message(
    message_data: MessageCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Send a message and get assistant response."""
    # Verify conversation belongs to user
    conversation = db.query(Conversation).filter(
        Conversation.id == message_data.conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Initialize metrics early
    if not conversation.metrics:
        conversation.metrics = {"kb_retrievals": [], "request_times": [], "total_turns": 0}
    conversation.metrics.setdefault("kb_retrievals", [])
    conversation.metrics.setdefault("request_times", [])
    
    # Pre-safety check (can be toggled via ENABLE_SAFETY)
    safety_verdict = "safe"
    safety_time = None
    if settings.ENABLE_SAFETY:
        try:
            safety_start = time.time()
            safety_verdict = classify_safety(message_data.content, username=current_user.username, conversation_id=conversation.id)
            safety_time = time.time() - safety_start
            if safety_verdict.startswith("unsafe"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="This content may be unsafe to discuss here. Please seek urgent in-person help or a trusted adult."
                )
            # Track safety check time
            if safety_time is not None:
                conversation.metrics["request_times"].append({
                    "type": "safety_check",
                    "time_seconds": round(safety_time, 3),
                    "timestamp": now_iso()
                })
        except HTTPException:
            raise
        except Exception as e:
            # Log error but continue
            write_detailed_log({
                "type": "safety_check_error",
                "timestamp": now_iso(),
                "error": str(e)
            }, username=current_user.username, conversation_id=conversation.id)
    
    # Get existing messages
    existing_messages = db.query(Message).filter(
        Message.conversation_id == conversation.id
    ).order_by(Message.sequence_number).all()
    
    # Convert to message format for LLM
    message_list = [
        {"role": msg.role.value, "content": msg.content}
        for msg in existing_messages
    ]
    
    # Add user message
    user_seq = len(existing_messages) + 1
    user_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.user,
        content=message_data.content,
        sequence_number=user_seq,
        extra_metadata={"safety_check": safety_verdict}
    )
    db.add(user_message)
    
    # Update conversation title from first user message if it's still "New Conversation" or None
    if not conversation.title or conversation.title == "New Conversation":
        # Use first 50 characters of the first user message as title
        title = message_data.content[:50].strip()
        if len(message_data.content) > 50:
            title += "..."
        conversation.title = title
    
    db.commit()
    
    # Write to conversation text log
    write_conversation_log(
        conversation_id=conversation.id,
        role="user",
        content=message_data.content,
        username=current_user.username
    )
    
    message_list.append({"role": "user", "content": message_data.content})
    
    # Load existing summaries from conversation
    existing_summaries = []
    if conversation.summary_note and isinstance(conversation.summary_note, dict):
        bullets = conversation.summary_note.get("bullets", [])
        if bullets:
            existing_summaries = bullets if isinstance(bullets, list) else [bullets]
    
    # Build messages with context
    llm_messages, context_info = build_messages(
        user=current_user,
        messages=message_list,
        diagnostic_slots=conversation.diagnostic_slots or {},
        summaries=existing_summaries,
        username=current_user.username,
        conversation_id=conversation.id
    )
    
    # Track KB retrieval in metrics
    if "kb_retrieval" in context_info:
        kb_info = context_info["kb_retrieval"]
        conversation.metrics["kb_retrievals"].append({
            "turn": conversation.metrics.get("total_turns", 0) + 1,
            "cards_returned": kb_info["cards_returned"],
            "method": kb_info["method"],
            "timestamp": now_iso(),
            "query": kb_info["query"]
        })
        # Track KB retrieval time
        conversation.metrics["request_times"].append({
            "type": "kb_retrieval",
            "time_seconds": kb_info["time_seconds"],
            "timestamp": now_iso()
        })
    
    # Track summarization time if it occurred
    if "summarize_time" in context_info:
        conversation.metrics["request_times"].append({
            "type": "summarize",
            "time_seconds": context_info["summarize_time"],
            "timestamp": now_iso()
        })
    
    # Save summaries if they were generated or updated
    if "summaries" in context_info:
        if not conversation.summary_note:
            conversation.summary_note = {}
        conversation.summary_note["bullets"] = context_info["summaries"]
        flag_modified(conversation, "summary_note")
    
    # Get assistant response
    chat_elapsed = None
    try:
        chat_start = time.time()
        assistant_response = chat_structured(llm_messages, username=current_user.username, conversation_id=conversation.id)
        chat_elapsed = time.time() - chat_start
    except Exception as e:
        # Still track time even on error
        if chat_elapsed is None:
            chat_elapsed = time.time() - chat_start if 'chat_start' in locals() else 0
        write_detailed_log({
            "type": "llm_error",
            "timestamp": now_iso(),
            "error": str(e),
            "conversation_id": conversation.id,
            "time_seconds": round(chat_elapsed, 3) if chat_elapsed else None
        }, username=current_user.username, conversation_id=conversation.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )
    
    # Update diagnostic slots
    if assistant_response.slotUpdates:
        conversation.diagnostic_slots = apply_slot_updates(
            conversation.diagnostic_slots or {},
            assistant_response.slotUpdates
        )
    
    # Update summary note
    if assistant_response.riskAssessment:
        ra = assistant_response.riskAssessment
        if not conversation.summary_note:
            conversation.summary_note = {}
        conversation.summary_note["riskLevel"] = ra.get("level", "none")
        if ra.get("reasons"):
            conversation.summary_note["hypotheses"] = ra.get("reasons", [])[:4]
        flag_modified(conversation, "summary_note")
    
    if assistant_response.nextSteps:
        ns = assistant_response.nextSteps
        if not conversation.summary_note:
            conversation.summary_note = {}
        if ns.get("selfHelp"):
            conversation.summary_note["nextSteps"] = ns.get("selfHelp", [])[:4]
        flag_modified(conversation, "summary_note")
    
    # Total turns (user-assistant pairs)
    conversation.metrics["total_turns"] = conversation.metrics.get("total_turns", 0) + 1

    # Track chat latency (seconds) for this turn
    if chat_elapsed is not None:
        conversation.metrics["request_times"].append({
            "type": "chat_structured",
            "time_seconds": round(chat_elapsed, 3),
            "timestamp": now_iso()
        })
    
    # Mark metrics as modified so SQLAlchemy detects the JSON field change
    flag_modified(conversation, "metrics")
    
    # Create assistant message
    assistant_content = f"{assistant_response.message}\n{assistant_response.question_next}".strip()
    assistant_seq = user_seq + 1
    assistant_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.assistant,
        content=assistant_content,
        sequence_number=assistant_seq,
        slot_updates=assistant_response.slotUpdates,
        risk_assessment=assistant_response.riskAssessment,
        next_steps=assistant_response.nextSteps,
        extra_metadata={"languageHint": assistant_response.languageHint}
    )
    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)
    
    # Write to conversation text log
    write_conversation_log(
        conversation_id=conversation.id,
        role="assistant",
        content=assistant_content,
        username=current_user.username
    )
    
    # Refresh conversation to get updated data and ensure messages are loaded
    db.refresh(conversation)
    # Ensure messages are loaded (they should be via relationship, but refresh to be sure)
    _ = conversation.messages  # Access to trigger lazy load if needed
    
    # Save conversation snapshot as JSON
    save_conversation_snapshot(
        conversation=conversation,
        user=current_user,
        username=current_user.username
    )
    
    return assistant_message

