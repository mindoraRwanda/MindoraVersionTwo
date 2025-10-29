# routers/messages.py
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from ..db import get_db
from ..services.safety_pipeline import handle_user_message_and_safety

router = APIRouter()

@router.post("/messages")
def post_message(payload: dict, background: BackgroundTasks, db: Session = Depends(get_db)):
    # payload expected: {user_id, conversation_id, content}
    return handle_user_message_and_safety(
        db=db,
        background=background,
        user_id=payload["user_id"],
        conversation_id=payload["conversation_id"],
        text=payload["content"],
    )
