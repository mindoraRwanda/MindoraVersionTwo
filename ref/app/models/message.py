"""Message model."""
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Enum as SQLEnum, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import enum
import uuid


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    slot_updates = Column(JSON, default=dict)  # Stores slot updates from assistant
    risk_assessment = Column(JSON, default=dict)  # Stores risk assessment
    next_steps = Column(JSON, default=dict)  # Stores next steps
    extra_metadata = Column(JSON, default=dict)  # Additional metadata (safety check, KB retrieval, etc.)
    sequence_number = Column(Integer, nullable=False)  # Order within conversation
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

