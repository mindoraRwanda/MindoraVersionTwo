"""
Session State Management for Stateful Conversations

This module manages conversation sessions with persistent state across multiple turns,
enabling finite state machine (FSM) based dialogue flows for mental health support.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta


class ConversationState(Enum):
    """Finite states for the conversation FSM"""
    INITIAL_DISTRESS = "initial_distress"
    AWAITING_ELABORATION = "awaiting_elaboration"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    SUGGESTION_PENDING = "suggestion_pending"
    GUIDING_IN_PROGRESS = "guiding_in_progress"
    CONVERSATION_IDLE = "conversation_idle"
    CRISIS_INTERVENTION = "crisis_intervention"


class CrisisSeverity(Enum):
    """Crisis severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SessionState:
    """Represents the current state of a conversation session"""
    session_id: str
    user_id: str
    current_state: ConversationState
    state_data: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    crisis_flags: List[str]
    crisis_severity: CrisisSeverity
    language_preference: str  # Language code (en, rw, fr, sw)
    created_at: datetime
    updated_at: datetime
    last_activity: datetime
    timeout_minutes: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert session state to dictionary for storage"""
        return {
            **asdict(self),
            'current_state': self.current_state.value,
            'crisis_severity': self.crisis_severity.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_activity': self.last_activity.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Create session state from dictionary"""
        return cls(
            session_id=data['session_id'],
            user_id=data['user_id'],
            current_state=ConversationState(data['current_state']),
            state_data=data.get('state_data', {}),
            conversation_history=data.get('conversation_history', []),
            crisis_flags=data.get('crisis_flags', []),
            crisis_severity=CrisisSeverity(data.get('crisis_severity', 'low')),
            language_preference=data.get('language_preference', 'en'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            last_activity=datetime.fromisoformat(data['last_activity']),
            timeout_minutes=data.get('timeout_minutes', 30)
        )

    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.now() - self.last_activity > timedelta(minutes=self.timeout_minutes)

    def update_activity(self):
        """Update last activity timestamp"""
        self.updated_at = datetime.now()
        self.last_activity = datetime.now()

    def add_to_history(self, message: Dict[str, Any]):
        """Add a message to conversation history"""
        self.conversation_history.append(message)
        self.update_activity()

    def set_crisis_flag(self, flag: str, severity: CrisisSeverity):
        """Set crisis flag and severity"""
        if flag not in self.crisis_flags:
            self.crisis_flags.append(flag)
        self.crisis_severity = severity
        self.update_activity()


class SessionStateManager:
    """Manages conversation sessions with persistent state"""

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        # Don't start cleanup task automatically to avoid event loop issues

    def start_cleanup_task(self):
        """Start background task to clean up expired sessions"""
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        except RuntimeError:
            # No running event loop, skip task creation for now
            pass

    def _start_cleanup_task(self):
        """Legacy method - use start_cleanup_task() instead"""
        # This method is kept for backward compatibility but doesn't actually start the task
        pass

    async def _cleanup_expired_sessions(self):
        """Background task to remove expired sessions"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                expired_sessions = [
                    session_id for session_id, session in self._sessions.items()
                    if session.is_expired()
                ]

                for session_id in expired_sessions:
                    del self._sessions[session_id]
                    print(f"Cleaned up expired session: {session_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)

    def create_session(self, user_id: str, initial_state: ConversationState = ConversationState.INITIAL_DISTRESS, language_preference: str = 'en') -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())

        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            current_state=initial_state,
            state_data={},
            conversation_history=[],
            crisis_flags=[],
            crisis_severity=CrisisSeverity.LOW,
            language_preference=language_preference,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_activity=datetime.now()
        )

        self._sessions[session_id] = session
        return session_id

    def create_session_with_id(self, session_id: str, user_id: str, initial_state: ConversationState = ConversationState.INITIAL_DISTRESS, language_preference: str = 'en') -> str:
        """Create a new conversation session with a specific session_id"""
        # Check if session already exists
        if session_id in self._sessions:
            existing_session = self._sessions[session_id]
            if not existing_session.is_expired():
                return session_id  # Return existing session ID if still active

        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            current_state=initial_state,
            state_data={},
            conversation_history=[],
            crisis_flags=[],
            crisis_severity=CrisisSeverity.LOW,
            language_preference=language_preference,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_activity=datetime.now()
        )

        self._sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID"""
        session = self._sessions.get(session_id)
        if session and not session.is_expired():
            return session
        elif session and session.is_expired():
            del self._sessions[session_id]
        return None

    def update_session_state(self, session_id: str, new_state: ConversationState,
                           state_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update session state"""
        session = self.get_session(session_id)
        if not session:
            return False

        session.current_state = new_state
        if state_data:
            session.state_data.update(state_data)
        session.update_activity()
        return True

    def update_session_language(self, session_id: str, language_preference: str) -> bool:
        """Update session language preference"""
        session = self.get_session(session_id)
        if not session:
            return False

        session.language_preference = language_preference
        session.update_activity()
        return True

    def add_message_to_history(self, session_id: str, role: str, content: str,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add message to session history"""
        session = self.get_session(session_id)
        if not session:
            return False

        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        session.add_to_history(message)
        return True

    def set_crisis_state(self, session_id: str, crisis_flags: List[str],
                        severity: CrisisSeverity) -> bool:
        """Set crisis state for session"""
        session = self.get_session(session_id)
        if not session:
            return False

        for flag in crisis_flags:
            session.set_crisis_flag(flag, severity)

        return True

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session summary for debugging/logging"""
        session = self.get_session(session_id)
        if not session:
            return None

        return {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'current_state': session.current_state.value,
            'crisis_severity': session.crisis_severity.value,
            'message_count': len(session.conversation_history),
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'is_expired': session.is_expired()
        }

    def get_active_sessions_count(self) -> int:
        """Get count of active (non-expired) sessions"""
        return len([s for s in self._sessions.values() if not s.is_expired()])

    def force_expire_session(self, session_id: str) -> bool:
        """Force expire a session (for testing or admin purposes)"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def save_session_to_storage(self, session_id: str) -> Optional[str]:
        """Save session to JSON string for external storage"""
        session = self.get_session(session_id)
        if not session:
            return None

        return json.dumps(session.to_dict(), indent=2)

    def load_session_from_storage(self, session_json: str, user_id: str) -> Optional[str]:
        """Load session from JSON string"""
        try:
            data = json.loads(session_json)
            data['user_id'] = user_id  # Ensure user_id matches current user
            session = SessionState.from_dict(data)

            # Only load if not expired
            if not session.is_expired():
                self._sessions[session.session_id] = session
                return session.session_id
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading session from storage: {e}")

        return None


# Global session manager instance
session_manager = SessionStateManager()