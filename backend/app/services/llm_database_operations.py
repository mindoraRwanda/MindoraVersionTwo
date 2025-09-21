"""
Database operations for the LLM service.
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.app.db.database import get_db
from backend.app.db.models import Message, Conversation
from .llm_config import MAX_CONVERSATION_HISTORY, MIN_MEANINGFUL_MESSAGE_LENGTH, ERROR_MESSAGES


class DatabaseManager:
    """Manages database operations for conversations and messages."""

    @staticmethod
    def fetch_recent_conversation(user_id: str, limit: int = MAX_CONVERSATION_HISTORY) -> List[Dict[str, str]]:
        """Fetch recent conversation history with enhanced context"""
        try:
            db: Session = next(get_db())
            convo = db.query(Conversation).filter_by(user_id=user_id).order_by(
                Conversation.last_activity_at.desc()
            ).first()

            if not convo:
                return []

            # Get messages for better context
            messages = db.query(Message).filter_by(conversation_id=convo.id).order_by(
                Message.timestamp.desc()
            ).limit(limit).all()

            conversation_messages = [
                {"role": m.sender, "text": m.content, "timestamp": m.timestamp}
                for m in reversed(messages)
            ]

            # Filter out very short or system messages for better context quality
            meaningful_messages = []
            for msg in conversation_messages:
                if len(msg["text"].strip()) >= MIN_MEANINGFUL_MESSAGE_LENGTH:
                    meaningful_messages.append({"role": msg["role"], "text": msg["text"]})

            return meaningful_messages
        except Exception as e:
            print(f"[DB Error] {e}")
            return []

    @staticmethod
    def get_conversation_context(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation context for a user"""
        try:
            db: Session = next(get_db())

            # Get most recently active conversation
            conversation = (
                db.query(Conversation)
                .filter_by(user_id=user_id)
                .order_by(Conversation.last_activity_at.desc())
                .first()
            )

            if not conversation:
                return []

            messages = (
                db.query(Message)
                .filter_by(conversation_id=conversation.id)
                .order_by(Message.timestamp.desc())
                .limit(limit)
                .all()
            )

            return list(reversed(messages))  # Return oldest → newest
        except Exception as e:
            print(f"Database error: {e}")
            return []