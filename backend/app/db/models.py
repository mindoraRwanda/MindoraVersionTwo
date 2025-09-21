from sqlalchemy import Column, Integer, Text, ForeignKey, TIMESTAMP, String
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(Text, nullable=False) 
    email = Column(Text, nullable=False, unique=True)  # Email remains unique
    password = Column(Text, nullable=False) 
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    conversations = relationship("Conversation", back_populates="user")

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    started_at = Column(TIMESTAMP, default=datetime.utcnow)
    last_activity_at = Column(TIMESTAMP, default=datetime.utcnow)  # added for tracking
    
    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    sender = Column(String(10), nullable=False)  # "user" or "bot"
    content = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")

class EmotionLog(Base):
    __tablename__ = "emotion_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"))
    input_text = Column(Text)
    detected_emotion = Column(String(50))
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)    



# models.py
# from sqlalchemy import (
#     Column, Integer, String, Text, ForeignKey, TIMESTAMP, Enum, Boolean, func,
#     UniqueConstraint
# )
# from sqlalchemy.orm import relationship
# import enum
# from .database import Base

# # --- existing enums ---
# class SenderType(enum.Enum):
#     user = "user"
#     bot = "bot"

# # ---------- NEW: Therapist ----------
# class Therapist(Base):
#     __tablename__ = "therapists"

#     id = Column(Integer, primary_key=True, index=True)
#     full_name = Column(String(150), nullable=False, index=True)
#     email = Column(String(255), nullable=False, unique=True, index=True)
#     phone = Column(String(50))
#     specialization = Column(String(120))
#     active = Column(Boolean, server_default="true", nullable=False)
#     created_at = Column(TIMESTAMP, server_default=func.now())

#     # patients assigned to this therapist (via association table)
#     patients = relationship(
#         "UserTherapist",
#         back_populates="therapist",
#         cascade="all, delete-orphan"
#     )

# # ---------- UPDATED: User (add link to therapists via association) ----------
# class User(Base):
#     __tablename__ = 'users'
#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String(100), nullable=False, unique=True, index=True)
#     email = Column(String(255), nullable=False, unique=True, index=True)
#     password = Column(String(255), nullable=False)  # store hash
#     created_at = Column(TIMESTAMP, server_default=func.now())

#     conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
#     emotion_logs = relationship("EmotionLog", back_populates="user", cascade="all, delete-orphan")

#     # therapists assigned to this user (via association table)
#     therapists = relationship(
#         "UserTherapist",
#         back_populates="user",
#         cascade="all, delete-orphan"
#     )

# # ---------- NEW: Association table (many-to-many + metadata) ----------
# class UserTherapist(Base):
#     """
#     Many-to-many assignment between users and therapists.
#     A user can have many therapists, a therapist can have many users.
#     """
#     __tablename__ = "user_therapists"

#     id = Column(Integer, primary_key=True)
#     user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
#     therapist_id = Column(Integer, ForeignKey("therapists.id", ondelete="CASCADE"), nullable=False, index=True)

#     # useful metadata
#     is_primary = Column(Boolean, server_default="false", nullable=False)
#     status = Column(String(30), server_default="active")   # active, paused, ended
#     assigned_at = Column(TIMESTAMP, server_default=func.now())

#     user = relationship("User", back_populates="therapists")
#     therapist = relationship("Therapist", back_populates="patients")

#     __table_args__ = (
#         UniqueConstraint("user_id", "therapist_id", name="uq_user_therapist_unique"),
#     )

# # ---------- existing conversation/message/emotion (unchanged from the improved version you liked) ----------
# class Conversation(Base):
#     __tablename__ = 'conversations'
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey('users.id', ondelete="CASCADE"), nullable=False, index=True)
#     started_at = Column(TIMESTAMP, server_default=func.now())
#     last_activity_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

#     user = relationship("User", back_populates="conversations")
#     messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
#     emotion_logs = relationship("EmotionLog", back_populates="conversation", cascade="all, delete-orphan")

# class Message(Base):
#     __tablename__ = 'messages'
#     id = Column(Integer, primary_key=True, index=True)
#     conversation_id = Column(Integer, ForeignKey('conversations.id', ondelete="CASCADE"), nullable=False, index=True)
#     sender = Column(Enum(SenderType), nullable=False)
#     content = Column(Text, nullable=False)
#     timestamp = Column(TIMESTAMP, server_default=func.now(), index=True)

#     conversation = relationship("Conversation", back_populates="messages")

# class EmotionLog(Base):
#     __tablename__ = "emotion_logs"
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
#     conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
#     input_text = Column(Text, nullable=False)
#     detected_emotion = Column(String(50), nullable=False, index=True)
#     timestamp = Column(TIMESTAMP, server_default=func.now(), index=True)

#     user = relationship("User", back_populates="emotion_logs")
#     conversation = relationship("Conversation", back_populates="emotion_logs")

