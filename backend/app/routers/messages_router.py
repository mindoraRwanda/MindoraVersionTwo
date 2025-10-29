# backend/app/routers/messages_router.py
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
import bleach
import time
import json
import os
from sqlalchemy.orm import Session
from typing import List, Dict, Any


from backend.app.auth.utils import get_current_user
from backend.app.db.database import SessionLocal
from backend.app.db.models import Conversation, Message, User, EmotionLog, SenderType
from backend.app.auth.schemas import MessageCreate, MessageOut
from backend.app.services.query_validator_langgraph import LangGraphQueryValidator
from backend.app.services.langgraph_state import QueryType

# Crisis classifier + pipeline (support either filename: crisis_ vs crises_)
try:
    from backend.app.services.crisis_classifier import classify_crisis
except ImportError:  # fallback if your file is named "crises_classifier.py"
    from backend.app.services.crises_classifier import classify_crisis

from backend.app.services.safety_pipeline import log_crisis_and_notify

router = APIRouter(prefix="/auth", tags=["Messages"])

# ---- Settings (env-driven) ----
CRISIS_CONFIDENCE_MIN = float(os.getenv("CRISIS_CONFIDENCE_MIN", "0.7"))
CRISIS_LABELS = {
    "self_harm", "suicide_ideation", "self_injury",
    "substance_abuse", "violence", "medical_emergency"
}

# Dependency: get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Global service getters (initialized in main.py)
def get_llm_service():
    from backend.app.main import llm_service
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service not initialized. Please try again later.")
    return llm_service

def get_query_validator():
    from backend.app.main import query_validator
    if not query_validator:
        raise HTTPException(status_code=503, detail="Query validator not initialized. Please try again later.")
    return query_validator


# --- Message Handling Endpoints ---

@router.post("/messages", response_model=MessageOut)
async def send_message(
    message: MessageCreate,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    llm_service = Depends(get_llm_service),
    query_validator: LangGraphQueryValidator = Depends(get_query_validator),
):
    """
    Pipeline:
      1) Verify conversation + clean input + fetch short history
      2) Run LangGraph validator (for routing + emotion context)
      3) Save USER message (flush → get id)
      4) ALWAYS run LLM crisis classifier; if crisis+confident → log + email
      5) Generate bot reply (override with crisis-safe message when needed)
      6) Save BOT message + EmotionLog; commit once
    """
    pipeline_start = time.time()
    print(f"\n🚀 Starting enhanced message pipeline for user {user.id}")

    # --- Verify conversation ownership ---
    db_start = time.time()
    convo = db.query(Conversation).filter_by(
        id=message.conversation_id,
        user_id=user.id
    ).first()
    db_lookup_time = time.time() - db_start
    print(f"⏱️  DB conversation lookup: {db_lookup_time:.3f}s")

    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # --- Clean input ---
    clean_start = time.time()
    clean_content = bleach.clean((message.content or "").strip())
    clean_time = time.time() - clean_start
    print(f"⏱️  Content cleaning: {clean_time:.3f}s")

    if not clean_content:
        raise HTTPException(status_code=400, detail="Message content is empty or invalid")

    # --- Load context window ---
    history_start = time.time()
    recent_history = (
        db.query(Message)
        .filter_by(conversation_id=message.conversation_id)
        .order_by(Message.timestamp.desc())
        .limit(15)
        .all()
    )
    recent_history.reverse()
    conversation_history = [
        {
            "role": (msg.sender.value if hasattr(msg.sender, "value") else msg.sender),
            "text": msg.content
        }
        for msg in recent_history
    ]
    history_time = time.time() - history_start
    print(f"⏱️  DB history load: {history_time:.3f}s ({len(recent_history)} messages)")

    # --- Step 1: LangGraph workflow (routing + optional emotion) ---
    validation_result = None
    try:
        validation_result = await query_validator.execute_workflow(
            clean_content,
            conversation_history=conversation_history
        )

        routing_decision = validation_result.get("routing_decision", "standard_processing")
        print(f"Query validation result: {json.dumps(validation_result, indent=2)}")

        if routing_decision == "crisis_intervention":
            # provisional reply (may be overridden by classifier below)
            bot_reply = await handle_crisis_query(clean_content, validation_result)

        elif routing_decision in ("mental_health_support", "standard_processing"):
            llm_start = time.time()
            emotion_data = validation_result.get("emotion_detection", {})
            bot_reply = await llm_service.generate_response(
                clean_content, conversation_history, str(user.id),
                skip_analysis=False, emotion_data=emotion_data
            )
            print(f"⏱️  LLM generation: {time.time() - llm_start:.3f}s ({len(bot_reply)} chars)")

        elif routing_decision == "random_question_filtered":
            bot_reply = await handle_random_question(clean_content, validation_result)

        elif routing_decision == "clarification_needed":
            bot_reply = await handle_unclear_query(clean_content, validation_result)

        else:
            bot_reply = validation_result.get(
                "final_response",
                "I'm here to help with mental health and emotional support. Could you please clarify what you'd like help with?"
            )

    except Exception as e:
        print(f"Query validation failed: {e}")
        llm_start = time.time()
        bot_reply = await llm_service.generate_response(
            clean_content, conversation_history, str(user.id)
        )
        print(f"⏱️  Fallback LLM generation: {time.time() - llm_start:.3f}s ({len(bot_reply)} chars)")

    # --- Emotion extraction from LangGraph (if any) ---
    emotion_detection = validation_result.get("emotion_detection", {}) if validation_result else {}
    detected_emotion = (
        emotion_detection.get("detected_emotion", "neutral")
        if isinstance(emotion_detection, dict) else "neutral"
    )

    # --- Step 2: Save USER message first (flush to get id) ---
    db_prep_start = time.time()
    user_msg = Message(
        conversation_id=message.conversation_id,
        sender=SenderType.user,   # use enum to match DB column
        content=clean_content
    )
    db.add(user_msg)
    db.flush()  # now we have user_msg.id

    # --- Step 3: ALWAYS run crisis classifier on every message ---
    try:
        crisis_result = classify_crisis(clean_content)  # {'label','severity','confidence','rationale'}
    except Exception as e:
        crisis_result = {
            "label": "other",
            "severity": "low",
            "confidence": 0.0,
            "rationale": f"classifier_error: {e}"
        }

    label = str(crisis_result.get("label", "other"))
    severity = str(crisis_result.get("severity", "low")).lower()
    confidence = float(crisis_result.get("confidence", 0.0))

    is_classifier_crisis = (label in CRISIS_LABELS) and (confidence >= CRISIS_CONFIDENCE_MIN)

    # --- Step 4: Log + email ONLY if truly crisis and confident enough ---
    did_notify = False
    if is_classifier_crisis:
        crisis_id = log_crisis_and_notify(
            db=db,
            background=background,
            user_id=user.id,
            conversation_id=message.conversation_id,
            message_id=user_msg.id,
            text=clean_content,
            crisis_result=crisis_result,
            classifier_model="llama3-8b-8192",
            classifier_version="v1",
        )
        did_notify = True
        print(f"🧾 Crisis logged: id={crisis_id} (label={label}, severity={severity}, conf={confidence:.2f})")

    # --- Step 5: If classifier says crisis, override reply with crisis-safe text (defense-in-depth) ---
    if did_notify:
        if severity in ("imminent", "high"):
            bot_reply = (
                "🚨 I’m worried about your safety. Your life has value, and you’re not alone.\n\n"
                "Please reach out right now:\n"
                "• Emergency Services: 112\n"
                "• Mental Health Helpline: 114 (24/7, free, confidential)\n"
                "• Ndera Neuropsychiatric Hospital: +250 781 447 928\n\n"
                "If you can, tell me where you are and if you’re safe. I’m here with you."
            )
        else:
            bot_reply = (
                "I’m concerned by what you shared. It’s important to talk with a professional.\n\n"
                "Available right now:\n"
                "• Mental Health Helpline: 114\n"
                "• Emergency Services: 112\n"
                "• Your nearest health center\n\n"
                "Would you like help connecting to these services?"
            )

    # --- Step 6: Save BOT message + EmotionLog; commit once ---
    bot_msg = Message(
        conversation_id=message.conversation_id,
        sender=SenderType.bot,   # use enum
        content=bleach.clean(bot_reply)
    )
    emotion_log = EmotionLog(
        user_id=user.id,
        conversation_id=message.conversation_id,
        input_text=clean_content,
        detected_emotion=detected_emotion
    )
    db.add_all([bot_msg, emotion_log])

    convo.last_activity_at = bot_msg.timestamp
    db.commit()

    db.refresh(bot_msg)
    db_save_time = time.time() - db_prep_start
    print(f"⏱️  DB save operations: {db_save_time:.3f}s")

    total_time = time.time() - pipeline_start
    print(f"🏁 Total pipeline time: {total_time:.3f}s")
    print(f"📊 Breakdown: DB({db_lookup_time + history_time + db_save_time:.3f}s) | "
          f"LLM/Validation({total_time - (db_lookup_time + history_time + db_save_time):.3f}s)")

    return bot_msg


@router.get("/context", response_model=List[MessageOut])
def get_context_window(
    limit: int = Query(default=10, le=50),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get context window of recent messages for the user."""
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
    return list(reversed(messages))  # oldest → newest


# --- Helpers for routed replies (non-crisis or provisional) ---

async def handle_crisis_query(query: str, validation_result: Dict[str, Any]) -> str:
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
    suggestions = validation_result.get("suggestions", [])
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
            "However, if this technical issue is causing you stress or affecting your mental health, "
            "I'm here to help you cope with those feelings. Would you like to talk about how this is impacting you emotionally?"
        )
    else:
        return (
            "I understand you have a question, but I'm primarily designed to support mental health and emotional well-being. "
            "While I can try to help with general questions, my expertise is in mental health support.\n\n"
            "Is there anything related to your mental health or well-being you'd like to discuss?"
        )

async def handle_unclear_query(query: str, validation_result: Dict[str, Any]) -> str:
    return (
        "I'm not sure I understand your query. Could you please clarify what you'd like help with?\n\n"
        "You can ask me about stress, mood, anxiety, relationships, or general mental health questions."
    )


# ADD THIS near the top of messages_router.py
async def process_clean_message(
    clean_content: str,
    conversation_id: int,
    background: BackgroundTasks,
    db: Session,
    user: User,
    llm_service,
    query_validator: LangGraphQueryValidator,
    input_modality: str = "text",
    input_meta: Dict[str, Any] | None = None,
):
    
    from fastapi import HTTPException
    from backend.app.db.models import Conversation, Message, EmotionLog, SenderType
    # from backend.app.services.safety_pipeline import log_crisis_and_notify
    try:
        from backend.app.services.crisis_classifier import classify_crisis
    except ImportError:
        from backend.app.services.crises_classifier import classify_crisis

    # 1) verify convo
    convo = db.query(Conversation).filter_by(id=conversation_id, user_id=user.id).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 2) short history
    recent = (
        db.query(Message)
        .filter_by(conversation_id=conversation_id)
        .order_by(Message.timestamp.desc())
        .limit(15)
        .all()
    )
    recent.reverse()
    conversation_history = [
        {"role": (m.sender.value if hasattr(m.sender, "value") else m.sender), "text": m.content}
        for m in recent
    ]

    # 3) validator / routing
    try:
        validation_result = await query_validator.execute_workflow(
            clean_content, conversation_history=conversation_history
        )
        routing_decision = validation_result.get("routing_decision", "standard_processing")
    except Exception as e:
        print(f"Validator error: {e}")
        validation_result, routing_decision = None, "fallback"

    # 4) save USER first (flush id)
    user_msg = Message(
        conversation_id=conversation_id,
        sender=SenderType.user,
        content=clean_content,
    )
    # tag voice/text + stt metadata if any
    user_msg.meta = (user_msg.meta or {})
    user_msg.meta["input_modality"] = input_modality
    if input_meta:
        user_msg.meta["input_meta"] = input_meta
    db.add(user_msg)
    db.flush()

    # 5) crisis classifier (always)
    try:
        crisis_result = classify_crisis(clean_content)
    except Exception as e:
        crisis_result = {"label": "other", "severity": "low", "confidence": 0.0, "rationale": f"classifier_error: {e}"}

    label = str(crisis_result.get("label", "other"))
    severity = str(crisis_result.get("severity", "low")).lower()
    confidence = float(crisis_result.get("confidence", 0.0))
    CRISIS_CONFIDENCE_MIN = float(os.getenv("CRISIS_CONFIDENCE_MIN", "0.7"))
    CRISIS_LABELS = {"self_harm", "suicide_ideation", "self_injury", "substance_abuse", "violence", "medical_emergency"}
    is_classifier_crisis = (label in CRISIS_LABELS) and (confidence >= CRISIS_CONFIDENCE_MIN)

    # 6) generate bot reply (your same branches)
    if routing_decision == "fallback":
        bot_reply = await llm_service.generate_response(clean_content, conversation_history, str(user.id))
    elif routing_decision == "random_question_filtered":
        bot_reply = await handle_random_question(clean_content, validation_result or {})
    elif routing_decision == "clarification_needed":
        bot_reply = await handle_unclear_query(clean_content, validation_result or {})
    elif routing_decision in ("mental_health_support", "standard_processing"):
        emotion_data = (validation_result or {}).get("emotion_detection", {})
        bot_reply = await llm_service.generate_response(
            clean_content, conversation_history, str(user.id), skip_analysis=False, emotion_data=emotion_data
        )
    elif routing_decision == "crisis_intervention":
        bot_reply = await handle_crisis_query(clean_content, validation_result or {})
    else:
        bot_reply = (validation_result or {}).get(
            "final_response", "I'm here to help with mental health support. Could you clarify what you need?"
        )

    # 7) crisis log & notify → override reply tone if needed
    did_notify = False
    if is_classifier_crisis:
        crisis_id = log_crisis_and_notify(
            db=db,
            background=background,
            user_id=user.id,
            conversation_id=conversation_id,
            message_id=user_msg.id,
            text=clean_content,
            crisis_result=crisis_result,
            classifier_model="llama3-8b-8192",
            classifier_version="v1",
        )
        did_notify = True
        print(f"🧾 Crisis logged: id={crisis_id} (label={label}, severity={severity}, conf={confidence:.2f})")

    if did_notify:
        if severity in ("imminent", "high"):
            bot_reply = (
                "🚨 I’m worried about your safety. Your life has value, and you’re not alone.\n\n"
                "Please reach out right now:\n"
                "• Emergency Services: 112\n"
                "• Mental Health Helpline: 114 (24/7, free, confidential)\n"
                "• Ndera Neuropsychiatric Hospital: +250 781 447 928\n\n"
                "If you can, tell me where you are and if you’re safe. I’m here with you."
            )
        else:
            bot_reply = (
                "I’m concerned by what you shared. It’s important to talk with a professional.\n\n"
                "Available right now:\n"
                "• Mental Health Helpline: 114\n"
                "• Emergency Services: 112\n"
                "• Your nearest health center\n\n"
                "Would you like help connecting to these services?"
            )

    # 8) save BOT + EmotionLog; commit once
    emotion_detection = (validation_result or {}).get("emotion_detection", {}) if validation_result else {}
    detected_emotion = emotion_detection.get("detected_emotion", "neutral") if isinstance(emotion_detection, dict) else "neutral"

    bot_msg = Message(conversation_id=conversation_id, sender=SenderType.bot, content=bleach.clean(bot_reply))
    emotion_log = EmotionLog(
        user_id=user.id,
        conversation_id=conversation_id,
        input_text=clean_content,
        detected_emotion=detected_emotion,
    )
    db.add_all([bot_msg, emotion_log])
    convo.last_activity_at = bot_msg.timestamp
    db.commit()
    db.refresh(bot_msg)
    return bot_msg

