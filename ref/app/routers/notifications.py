"""Notification endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from datetime import datetime
from app.database import get_db
from app.core.security import get_current_active_user
from app.models.user import User
from app.models.notification import Notification, NotificationStatus
from app.schemas.notification import NotificationCreate, NotificationResponse, NotificationListResponse

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.post("", response_model=NotificationResponse, status_code=status.HTTP_201_CREATED)
def create_notification(
    notification_data: NotificationCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a notification (admin or self)."""
    notification = Notification(
        user_id=current_user.id,
        title=notification_data.title,
        content=notification_data.content,
        notification_type=notification_data.notification_type,
        extra_metadata=notification_data.metadata or {},
        status=NotificationStatus.pending
    )
    db.add(notification)
    db.commit()
    db.refresh(notification)
    
    # In production, you would send the notification here (email, push, etc.)
    # For now, mark as sent
    notification.status = NotificationStatus.sent
    notification.sent_at = datetime.utcnow()
    db.commit()
    db.refresh(notification)
    
    return notification


@router.get("", response_model=NotificationListResponse)
def get_notifications(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    status_filter: NotificationStatus = Query(None, description="Filter by status"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's notifications."""
    query = db.query(Notification).filter(Notification.user_id == current_user.id)
    
    if status_filter:
        query = query.filter(Notification.status == status_filter)
    
    notifications = query.order_by(Notification.created_at.desc()).offset(skip).limit(limit).all()
    total = query.count()
    
    return {
        "notifications": notifications,
        "total": total
    }


@router.patch("/{notification_id}/read", response_model=NotificationResponse)
def mark_notification_read(
    notification_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mark a notification as read."""
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == current_user.id
    ).first()
    
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found"
        )
    
    notification.status = NotificationStatus.read
    notification.read_at = datetime.utcnow()
    db.commit()
    db.refresh(notification)
    
    return notification

