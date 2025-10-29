#voice_router.py
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
import os, uuid, tempfile, subprocess
from faster_whisper import WhisperModel

from backend.app.auth.utils import get_current_user
from backend.app.db.database import SessionLocal
from backend.app.db.models import User
from backend.app.services.query_validator_langgraph import LangGraphQueryValidator
from backend.app.routers.messages_router import process_clean_message

router = APIRouter(prefix="/voice", tags=["Voice"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_llm_service():
    from backend.app.main import llm_service
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service not initialized.")
    return llm_service

def get_query_validator():
    from backend.app.main import query_validator
    if not query_validator:
        raise HTTPException(status_code=503, detail="Query validator not initialized.")
    return query_validator

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

@router.post("/messages")
async def voice_message(
    background: BackgroundTasks,
    conversation_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    llm_service = Depends(get_llm_service),
    query_validator: LangGraphQueryValidator = Depends(get_query_validator),
):
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

    input_meta = {
        "stt_model": f"faster-whisper:{MODEL_SIZE}",
        "stt_device": DEVICE,
        "duration": getattr(info, "duration", None),
        "language": getattr(info, "language", None),
        "mime": file.content_type,
        "filename": file.filename,
    }

    # hand off to the same pipeline
    return await process_clean_message(
        clean_content=text,
        conversation_id=conversation_id,
        background=background,
        db=db,
        user=user,
        llm_service=llm_service,
        query_validator=query_validator,
        input_modality="voice",
        input_meta=input_meta,
    )
