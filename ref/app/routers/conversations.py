"""Conversation endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.core.security import get_current_active_user
from app.models.user import User
from app.models.conversation import Conversation
from app.schemas.conversation import ConversationCreate, ConversationResponse, ConversationListResponse

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=ConversationListResponse)
def get_conversations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's conversations."""
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.updated_at.desc()).offset(skip).limit(limit).all()
    
    # Add message count to each conversation
    conversations_with_count = []
    for conv in conversations:
        message_count = len(conv.messages)
        conv_dict = {
            **{k: v for k, v in conv.__dict__.items() if not k.startswith("_")},
            "message_count": message_count
        }
        conversations_with_count.append(ConversationResponse(**conv_dict))
    
    total = db.query(Conversation).filter(Conversation.user_id == current_user.id).count()
    
    return {
        "conversations": conversations_with_count,
        "total": total
    }


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    conv_data: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new conversation."""
    from app.services.slots import get_default_slots
    
    conversation = Conversation(
        user_id=current_user.id,
        title=conv_data.title,
        summary_note={"bullets": [], "riskLevel": "none", "nextSteps": [], "hypotheses": []},
        diagnostic_slots=get_default_slots(),
        metrics={"kb_retrievals": [], "request_times": [], "total_turns": 0}
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get conversation by ID."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Add message count
    message_count = len(conversation.messages)
    conv_dict = {
        **{k: v for k, v in conversation.__dict__.items() if not k.startswith("_")},
        "message_count": message_count
    }
    return ConversationResponse(**conv_dict)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    db.delete(conversation)
    db.commit()
    return None

