#voice_router.py
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
import os, uuid, tempfile, subprocess
from faster_whisper import WhisperModel
import time
import bleach
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from gtts import gTTS
import base64
from io import BytesIO

from ..auth.utils import get_current_user
from ..db.database import SessionLocal
from ..db.models import User, Conversation, Message, EmotionLog
from ..auth.schemas import UserOut
from ..services.session_state_manager import session_manager
from ..dependencies import get_stateful_pipeline
import logging  

logger = logging.getLogger("voice_router")

router = APIRouter(prefix="/voice", tags=["Voice"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# lazy load a single Whisper model
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

def _to_wav_16k_mono(src_path: str, dst_path: str):
    cmd = ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", "-f", "wav", dst_path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="ignore")[:2000])

@router.post("/messages", response_model=None)
async def voice_message(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    db: Session = Depends(get_db),
    user: UserOut = Depends(get_current_user),
):
    stateful_pipeline = get_stateful_pipeline(db=db, background=background)
    # --- Start of transcription logic ---
    # save upload
    suffix = os.path.splitext(file.filename or "")[-1] or ".webm"
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, f"in_{uuid.uuid4().hex}{suffix}")
        wav = os.path.join(td, f"pcm16_{uuid.uuid4().hex}.wav")
        data = await file.read()
        with open(raw, "wb") as f:
            f.write(data)

        # transcode → 16k mono wav
        try:
            _to_wav_16k_mono(raw, wav)
        except Exception as e:
            raise HTTPException(400, f"Audio decode failed: {e}")

        # transcribe
        try:
            segments, info = whisper_model.transcribe(
                wav, language=None, vad_filter=True, beam_size=5, temperature=0.0
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:
            raise HTTPException(500, f"Transcription failed: {e}")

    if not text:
        raise HTTPException(400, "No speech recognized")
    # --- End of transcription logic ---

    pipeline_start = time.time()
    print(f"\n🚀 Starting enhanced message pipeline for user {user.id} from voice input")

    # Single query to verify conversation ownership
    db_start = time.time()
    convo = db.query(Conversation).filter_by(
        uuid=conversation_id,
        user_id=user.id
    ).first()
    db_lookup_time = time.time() - db_start
    print(f"⏱️  DB conversation lookup: {db_lookup_time:.3f}s")

    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    clean_content = text # From transcription

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

    # Save user message to database BEFORE pipeline processing
    user_msg = Message(
        conversation_id=convo.id,
        sender="user",
        content=clean_content
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    print(f"💾 User message saved with ID: {user_msg.id}")

    # Use the stateful mental health pipeline
    workflow_start = time.time()
    pipeline_result = {}

    try:
        pipeline_result = await stateful_pipeline.process_query(
            query=clean_content,
            user_id=str(user.id),
            conversation_id=str(convo.id),
            message_id=str(user_msg.id),
            conversation_history=conversation_history,
            user_gender=str(user.gender),
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
        bot_reply = "I'm here to support you. How can I help you today?"
        workflow_time = time.time() - workflow_start
        print(f"⏱️  Fallback processing: {workflow_time:.3f}s")

    emotion_detection = pipeline_result.get("emotion_detection")
    detected_emotion = emotion_detection.selected_emotion if emotion_detection and hasattr(emotion_detection, 'selected_emotion') else "neutral"
    
    query_evaluation = pipeline_result.get("query_evaluation")
    routing_decision = query_evaluation.evaluation_type.value if query_evaluation and hasattr(query_evaluation, 'evaluation_type') else "GIVE_EMPATHY"

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

    db.add_all([bot_msg, emotion_log])
    convo.last_activity_at = bot_msg.timestamp
    db.commit()

    db.refresh(bot_msg)
    db_save_time = time.time() - db_prep_start
    print(f"⏱️  DB save operations: {db_save_time:.3f}s")

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

    total_time = time.time() - pipeline_start
    print(f"🏁 Total pipeline time: {total_time:.3f}s")

    # --- Text-to-Speech ---
    tts_start = time.time()
    audio_bytes = None
    try:
        tts = gTTS(text=bot_reply, lang='en', slow=False)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_bytes = mp3_fp.read()
    except Exception as e:
        print(f"TTS generation failed: {e}")

    tts_time = time.time() - tts_start
    print(f"⏱️  TTS generation: {tts_time:.3f}s")

    return {
        "id": bot_msg.uuid,
        "sender": bot_msg.sender.value,
        "content": bot_msg.content,
        "timestamp": bot_msg.timestamp,
        "audio_content": base64.b64encode(audio_bytes).decode('utf-8') if audio_bytes else None
    }