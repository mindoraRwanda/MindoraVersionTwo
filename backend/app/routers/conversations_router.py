from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from backend.app.auth.utils import get_current_user
from backend.app.db.database import SessionLocal
from backend.app.db.models import Conversation, Message, User, EmotionLog
from backend.app.auth.schemas import ConversationOut

router = APIRouter(prefix="/auth", tags=["Conversations"])

# Dependency: get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Conversation Management Endpoints ---

@router.post("/conversations", response_model=ConversationOut)
def create_conversation(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Create a new conversation for the authenticated user."""
    new_convo = Conversation(user_id=user.id)
    db.add(new_convo)
    db.commit()
    db.refresh(new_convo)
    return new_convo


@router.get("/conversations")
def list_user_conversations(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """List all conversations for the authenticated user."""
    conversations = db.query(Conversation).filter_by(user_id=user.id).order_by(Conversation.started_at.desc()).all()
    return [
        {
            "id": c.id,
            "started_at": c.started_at,
            "messages": [
                {
                    "id": msg.id,
                    "content": msg.content,
                    "sender": msg.sender,
                    "timestamp": msg.timestamp
                } for msg in db.query(Message).filter_by(conversation_id=c.id).order_by(Message.timestamp.asc()).limit(1).all()
            ]
        } for c in conversations
    ]


@router.get("/conversations/{conversation_id}/messages", response_model=List[dict])
def get_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Get all messages for a specific conversation."""
    conversation = db.query(Conversation)\
        .filter_by(id=conversation_id, user_id=user.id).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = db.query(Message)\
        .filter_by(conversation_id=conversation_id)\
        .order_by(Message.timestamp)\
        .all()

    return [
        {
            "id": msg.id,
            "content": msg.content,
            "sender": msg.sender,
            "timestamp": msg.timestamp
        } for msg in messages
    ]


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Delete a conversation and all its messages."""
    conversation = db.query(Conversation)\
        .filter_by(id=conversation_id, user_id=user.id).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Delete related emotion logs first to avoid foreign key constraint violation
    db.query(EmotionLog)\
        .filter_by(conversation_id=conversation_id)\
        .delete()

    # Now delete the conversation (messages will be cascade deleted due to relationship config)
    db.delete(conversation)
    db.commit()

    return {"message": "Conversation deleted successfully"}