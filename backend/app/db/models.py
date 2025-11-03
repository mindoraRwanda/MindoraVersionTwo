from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey, TIMESTAMP, Enum, Boolean, Float, func, UniqueConstraint, JSON
)
from sqlalchemy.orm import relationship
from .database import Base
import enum

# ----- enums -----
class SenderType(enum.Enum):
    user = "user"
    bot = "bot"

class CrisisType(enum.Enum):
    self_harm = "self_harm"
    suicide_ideation = "suicide_ideation"
    self_injury = "self_injury"
    substance_abuse = "substance_abuse"
    violence = "violence"
    medical_emergency = "medical_emergency"
    other = "other"

class CrisisSeverity(enum.Enum):
    low = "low"
    moderate = "moderate"
    high = "high"
    imminent = "imminent"

class CrisisStatus(enum.Enum):
    new = "new"
    notified = "notified"
    acknowledged = "acknowledged"
    resolved = "resolved"
    false_positive = "false_positive"

# ----- therapist & assignment -----
class Therapist(Base):
    __tablename__ = "therapists"
    id = Column(Integer, primary_key=True)
    full_name = Column(String(150), nullable=False, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    phone = Column(String(50))
    specialization = Column(String(120))
    active = Column(Boolean, server_default="true", nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())

    patients = relationship("UserTherapist", back_populates="therapist", cascade="all, delete-orphan")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), nullable=False, unique=True, index=True)
    phone = Column(String(50))
    email = Column(String(255), nullable=False, unique=True, index=True)
    password = Column(String(255), nullable=False)  # store hash
    gender = Column(String(20), nullable=True)  # Optional gender field for cultural personalization
    
    created_at = Column(TIMESTAMP, server_default=func.now())

    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    emotion_logs = relationship("EmotionLog", back_populates="user", cascade="all, delete-orphan")
    therapist_links = relationship("UserTherapist", back_populates="user", cascade="all, delete-orphan")
    crisis_logs = relationship("CrisisLog", back_populates="user")

class UserTherapist(Base):
    __tablename__ = "user_therapists"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    therapist_id = Column(Integer, ForeignKey("therapists.id", ondelete="CASCADE"), nullable=False, index=True)
    is_primary = Column(Boolean, server_default="false", nullable=False)
    status = Column(String(30), server_default="active")  # active, paused, ended
    assigned_at = Column(TIMESTAMP, server_default=func.now())

    user = relationship("User", back_populates="therapist_links")
    therapist = relationship("Therapist", back_populates="patients")
    __table_args__ = (UniqueConstraint("user_id", "therapist_id", name="uq_user_therapist_unique"),)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    started_at = Column(TIMESTAMP, server_default=func.now())
    last_activity_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    # Optional: store small per-convo metadata
    meta = Column(JSON, server_default="{}")

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    emotion_logs = relationship("EmotionLog", back_populates="conversation", cascade="all, delete-orphan")
    crisis_logs = relationship("CrisisLog", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    sender = Column(Enum(SenderType), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP, server_default=func.now(), index=True)
    # Flexible per-message metadata (e.g., safety flags, embeddings ids, etc.)
    meta = Column(JSON, server_default="{}")

    conversation = relationship("Conversation", back_populates="messages")

class EmotionLog(Base):
    __tablename__ = "emotion_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    input_text = Column(Text, nullable=False)
    detected_emotion = Column(String(50), nullable=False, index=True)
    timestamp = Column(TIMESTAMP, server_default=func.now(), index=True)

    user = relationship("User", back_populates="emotion_logs")
    conversation = relationship("Conversation", back_populates="emotion_logs")

# ----- NEW: crisis logs -----
class CrisisLog(Base):
    __tablename__ = "crisis_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="SET NULL"))
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="SET NULL"))

    detected_type = Column(Enum(CrisisType), nullable=False, index=True)
    severity = Column(Enum(CrisisSeverity), nullable=False, index=True)
    confidence = Column(Float, nullable=False)  # 0..1

    # snapshot for audit & triage
    excerpt = Column(Text, nullable=False)          # short snippet from user's message
    rationale = Column(Text, nullable=True)         # LLM rationale (optional)
    classifier_model = Column(String(120), nullable=False)   # e.g., "llama3-8b-8192"
    classifier_version = Column(String(40), nullable=True)

    status = Column(Enum(CrisisStatus), nullable=False, server_default=CrisisStatus.new.value, index=True)
    notified_therapist_id = Column(Integer, ForeignKey("therapists.id", ondelete="SET NULL"))
    notified_at = Column(TIMESTAMP, nullable=True)
    acknowledged_at = Column(TIMESTAMP, nullable=True)
    resolved_at = Column(TIMESTAMP, nullable=True)

    extra = Column(JSON, server_default="{}")  # arbitrary extra fields
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)

    user = relationship("User", back_populates="crisis_logs")
    conversation = relationship("Conversation", back_populates="crisis_logs")
