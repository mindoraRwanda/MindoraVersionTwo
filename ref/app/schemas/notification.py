"""Notification schemas."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.models.notification import NotificationStatus


class NotificationCreate(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    notification_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class NotificationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    content: str
    status: NotificationStatus
    notification_type: Optional[str]
    metadata: Dict[str, Any] = Field(alias="extra_metadata")
    created_at: datetime
    sent_at: Optional[datetime]
    read_at: Optional[datetime]
    
    class Config:
        from_attributes = True
        populate_by_name = True


class NotificationListResponse(BaseModel):
    notifications: List[NotificationResponse]
    total: int

