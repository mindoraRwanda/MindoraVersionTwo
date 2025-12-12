"""Message schemas."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from app.models.message import MessageRole


class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)
    conversation_id: str


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    slot_updates: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    next_steps: Dict[str, Any]
    metadata: Dict[str, Any] = Field(alias="extra_metadata")
    sequence_number: int
    created_at: datetime
    
    class Config:
        from_attributes = True
        populate_by_name = True


class MessageListResponse(BaseModel):
    messages: list[MessageResponse]
    total: int
    conversation_id: str

