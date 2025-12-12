"""Conversation schemas."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    summary_note: Dict[str, Any]
    diagnostic_slots: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime]
    message_count: Optional[int] = 0
    
    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    conversations: List[ConversationResponse]
    total: int

