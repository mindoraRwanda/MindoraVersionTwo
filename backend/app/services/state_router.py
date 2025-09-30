"""
State Router for Conversational Finite State Machine

This module implements the core FSM logic for managing conversation flow states
and transitions in the mental health chatbot system.
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass

from backend.app.services.session_state_manager import (
    SessionStateManager, ConversationState, CrisisSeverity, session_manager
)
from backend.app.services.crisis_interceptor import crisis_interceptor


class TransitionCondition(Enum):
    """Conditions that trigger state transitions"""
    USER_EXPRESSED_DISTRESS = "user_expressed_distress"
    USER_PROVIDED_DETAILS = "user_provided_details"
    USER_ASKED_QUESTION = "user_asked_question"
    USER_REQUESTED_HELP = "user_requested_help"
    USER_SEEKING_CLARITY = "user_seeking_clarity"
    CONVERSATION_IDLE = "conversation_idle"
    CRISIS_DETECTED = "crisis_detected"
    CRISIS_RESOLVED = "crisis_resolved"
    GUIDANCE_COMPLETED = "guidance_completed"
    USER_DISENGAGED = "user_disengaged"


@dataclass
class StateTransition:
    """Defines a state transition rule"""
    from_state: ConversationState
    to_state: ConversationState
    conditions: List[TransitionCondition]
    priority: int = 1  # Higher priority transitions are checked first


class StateRouter:
    """Finite State Machine router for conversation flow"""

    def __init__(self):
        self.transition_rules = self._initialize_transition_rules()

    def _initialize_transition_rules(self) -> List[StateTransition]:
        """Initialize FSM transition rules"""
        return [
            # Crisis intervention has highest priority
            StateTransition(
                from_state=ConversationState.INITIAL_DISTRESS,
                to_state=ConversationState.CRISIS_INTERVENTION,
                conditions=[TransitionCondition.CRISIS_DETECTED],
                priority=10
            ),
            StateTransition(
                from_state=ConversationState.AWAITING_ELABORATION,
                to_state=ConversationState.CRISIS_INTERVENTION,
                conditions=[TransitionCondition.CRISIS_DETECTED],
                priority=10
            ),
            StateTransition(
                from_state=ConversationState.AWAITING_CLARIFICATION,
                to_state=ConversationState.CRISIS_INTERVENTION,
                conditions=[TransitionCondition.CRISIS_DETECTED],
                priority=10
            ),
            StateTransition(
                from_state=ConversationState.SUGGESTION_PENDING,
                to_state=ConversationState.CRISIS_INTERVENTION,
                conditions=[TransitionCondition.CRISIS_DETECTED],
                priority=10
            ),
            StateTransition(
                from_state=ConversationState.GUIDING_IN_PROGRESS,
                to_state=ConversationState.CRISIS_INTERVENTION,
                conditions=[TransitionCondition.CRISIS_DETECTED],
                priority=10
            ),

            # Normal conversation flow transitions
            StateTransition(
                from_state=ConversationState.INITIAL_DISTRESS,
                to_state=ConversationState.AWAITING_ELABORATION,
                conditions=[TransitionCondition.USER_PROVIDED_DETAILS],
                priority=5
            ),
            StateTransition(
                from_state=ConversationState.AWAITING_ELABORATION,
                to_state=ConversationState.AWAITING_CLARIFICATION,
                conditions=[TransitionCondition.USER_ASKED_QUESTION],
                priority=5
            ),
            StateTransition(
                from_state=ConversationState.AWAITING_ELABORATION,
                to_state=ConversationState.SUGGESTION_PENDING,
                conditions=[TransitionCondition.USER_REQUESTED_HELP],
                priority=5
            ),
            StateTransition(
                from_state=ConversationState.AWAITING_CLARIFICATION,
                to_state=ConversationState.GUIDING_IN_PROGRESS,
                conditions=[TransitionCondition.USER_SEEKING_CLARITY],
                priority=5
            ),
            StateTransition(
                from_state=ConversationState.SUGGESTION_PENDING,
                to_state=ConversationState.GUIDING_IN_PROGRESS,
                conditions=[TransitionCondition.USER_PROVIDED_DETAILS],
                priority=5
            ),

            # Return to appropriate state after crisis
            StateTransition(
                from_state=ConversationState.CRISIS_INTERVENTION,
                to_state=ConversationState.AWAITING_ELABORATION,
                conditions=[TransitionCondition.CRISIS_RESOLVED],
                priority=8
            ),

            # Idle state transitions
            StateTransition(
                from_state=ConversationState.GUIDING_IN_PROGRESS,
                to_state=ConversationState.CONVERSATION_IDLE,
                conditions=[TransitionCondition.CONVERSATION_IDLE],
                priority=3
            ),
            StateTransition(
                from_state=ConversationState.CONVERSATION_IDLE,
                to_state=ConversationState.AWAITING_ELABORATION,
                conditions=[TransitionCondition.USER_EXPRESSED_DISTRESS],
                priority=5
            ),

            # Disengagement handling
            StateTransition(
                from_state=ConversationState.CONVERSATION_IDLE,
                to_state=ConversationState.INITIAL_DISTRESS,
                conditions=[TransitionCondition.USER_DISENGAGED],
                priority=2
            ),
        ]

    def determine_next_state(self, session_id: str, user_message: str,
                           emotion_data: Optional[Dict[str, Any]] = None) -> ConversationState:
        """
        Determine the next conversation state based on current state and user input

        Returns the next state for the conversation
        """
        current_session = session_manager.get_session(session_id)
        if not current_session:
            # Create new session if none exists
            session_id = session_manager.create_session("user")
            current_session = session_manager.get_session(session_id)

        if current_session is None:
            # Fallback if session creation failed
            return ConversationState.INITIAL_DISTRESS

        current_state = current_session.current_state

        # Check for crisis first (highest priority)
        is_crisis, _, _ = crisis_interceptor.detect_crisis(user_message, emotion_data)
        if is_crisis:
            return ConversationState.CRISIS_INTERVENTION

        # Check if currently in crisis state
        if current_state == ConversationState.CRISIS_INTERVENTION:
            # Check if crisis appears resolved
            if self._is_crisis_resolved(user_message, current_session):
                return ConversationState.AWAITING_ELABORATION
            return current_state  # Stay in crisis intervention

        # Get applicable transitions for current state
        applicable_transitions = [
            rule for rule in self.transition_rules
            if rule.from_state == current_state and rule.priority < 10  # Exclude crisis transitions
        ]

        # Sort by priority (highest first)
        applicable_transitions.sort(key=lambda x: x.priority, reverse=True)

        # Check each transition condition
        for transition in applicable_transitions:
            if self._check_transition_conditions(transition.conditions, user_message,
                                               current_session, emotion_data):
                return transition.to_state

        # Default: stay in current state if no transitions apply
        return current_state

    def _check_transition_conditions(self, conditions: List[TransitionCondition],
                                   user_message: str, session: Any,
                                   emotion_data: Optional[Dict[str, Any]] = None) -> bool:
        """Check if transition conditions are met"""
        for condition in conditions:
            if not self._evaluate_condition(condition, user_message, session, emotion_data):
                return False
        return True

    def _evaluate_condition(self, condition: TransitionCondition, user_message: str,
                          session: Any, emotion_data: Optional[Dict[str, Any]] = None) -> bool:
        """Evaluate a specific transition condition"""
        if condition == TransitionCondition.USER_EXPRESSED_DISTRESS:
            return self._detect_distress_expression(user_message)
        elif condition == TransitionCondition.USER_PROVIDED_DETAILS:
            return self._detect_detail_provided(user_message)
        elif condition == TransitionCondition.USER_ASKED_QUESTION:
            return self._detect_question_asked(user_message)
        elif condition == TransitionCondition.USER_REQUESTED_HELP:
            return self._detect_help_request(user_message)
        elif condition == TransitionCondition.USER_SEEKING_CLARITY:
            return self._detect_clarity_seeking(user_message)
        elif condition == TransitionCondition.CONVERSATION_IDLE:
            return self._detect_idle_conversation(session)
        elif condition == TransitionCondition.CRISIS_DETECTED:
            return crisis_interceptor.detect_crisis(user_message, emotion_data)[0]
        elif condition == TransitionCondition.CRISIS_RESOLVED:
            return self._is_crisis_resolved(user_message, session)
        elif condition == TransitionCondition.GUIDANCE_COMPLETED:
            return self._is_guidance_completed(session)
        elif condition == TransitionCondition.USER_DISENGAGED:
            return self._detect_user_disengagement(user_message)

        return False

    def _detect_distress_expression(self, message: str) -> bool:
        """Detect if user is expressing distress"""
        distress_keywords = [
            "struggling", "difficult", "hard", "overwhelmed", "stressed",
            "anxious", "worried", "sad", "depressed", "hurt", "pain"
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in distress_keywords)

    def _detect_detail_provided(self, message: str) -> bool:
        """Detect if user provided specific details"""
        # Check for specific situations, feelings, or contexts
        detail_indicators = [
            "because", "when", "after", "during", "before", "while",
            "feeling", "thinking", "situation", "problem", "issue"
        ]
        message_lower = message.lower()
        return (any(indicator in message_lower for indicator in detail_indicators) and
                len(message.split()) > 5)

    def _detect_question_asked(self, message: str) -> bool:
        """Detect if user asked a question"""
        return message.strip().endswith('?') or message.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who'))

    def _detect_help_request(self, message: str) -> bool:
        """Detect if user is requesting help"""
        help_keywords = [
            "help", "support", "advice", "guidance", "suggestion",
            "recommendation", "assist", "cope", "deal with", "handle"
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in help_keywords)

    def _detect_clarity_seeking(self, message: str) -> bool:
        """Detect if user is seeking clarity or understanding"""
        clarity_keywords = [
            "confused", "understand", "clarify", "explain", "mean",
            "example", "clear", "sure", "certain", "know"
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in clarity_keywords)

    def _detect_idle_conversation(self, session: Any) -> bool:
        """Detect if conversation has become idle"""
        # Consider idle if no messages for 5+ minutes or very short responses
        if not session.conversation_history:
            return False

        last_messages = session.conversation_history[-3:] if len(session.conversation_history) >= 3 else session.conversation_history
        return all(len(msg.get('content', '')) < 10 for msg in last_messages)

    def _is_crisis_resolved(self, message: str, session: Any) -> bool:
        """Check if crisis state appears resolved"""
        if not session.crisis_flags:
            return True

        # Check for positive indicators
        positive_keywords = [
            "better", "improving", "okay", "fine", "good",
            "thank you", "appreciate", "helpful", "support"
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in positive_keywords)

    def _is_guidance_completed(self, session: Any) -> bool:
        """Check if guidance phase is completed"""
        # Consider guidance complete if user has received multiple suggestions
        # and conversation has progressed significantly
        assistant_messages = [msg for msg in session.conversation_history
                            if msg.get('role') == 'assistant']
        return len(assistant_messages) >= 3

    def _detect_user_disengagement(self, message: str) -> bool:
        """Detect if user appears disengaged"""
        disengagement_keywords = [
            "whatever", "don't care", "not interested", "bye",
            "goodbye", "see you", "talk later", "end"
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in disengagement_keywords)

    def get_state_prompt_context(self, state: ConversationState) -> Dict[str, Any]:
        """Get prompt context for current state"""
        context_map = {
            ConversationState.INITIAL_DISTRESS: {
                "state_type": "empathy_building",
                "focus": "acknowledge_feelings",
                "response_style": "supportive_listening"
            },
            ConversationState.AWAITING_ELABORATION: {
                "state_type": "information_gathering",
                "focus": "encourage_details",
                "response_style": "open_ended_questions"
            },
            ConversationState.AWAITING_CLARIFICATION: {
                "state_type": "clarification",
                "focus": "seek_understanding",
                "response_style": "specific_questions"
            },
            ConversationState.SUGGESTION_PENDING: {
                "state_type": "solution_offering",
                "focus": "provide_options",
                "response_style": "practical_suggestions"
            },
            ConversationState.GUIDING_IN_PROGRESS: {
                "state_type": "active_guidance",
                "focus": "step_by_step_support",
                "response_style": "structured_guidance"
            },
            ConversationState.CONVERSATION_IDLE: {
                "state_type": "maintenance",
                "focus": "re_engagement",
                "response_style": "gentle_check_in"
            },
            ConversationState.CRISIS_INTERVENTION: {
                "state_type": "crisis_support",
                "focus": "immediate_safety",
                "response_style": "crisis_protocol"
            }
        }

        return context_map.get(state, context_map[ConversationState.INITIAL_DISTRESS])

    def should_continue_conversation(self, session_id: str) -> bool:
        """Determine if conversation should continue or end"""
        session = session_manager.get_session(session_id)
        if not session:
            return False

        # End conversation if user is disengaged or session expired
        if (session.current_state == ConversationState.CONVERSATION_IDLE and
            self._detect_user_disengagement("") and
            len(session.conversation_history) > 10):
            return False

        return not session.is_expired()

    def get_conversation_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of conversation state for logging/debugging"""
        session = session_manager.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "current_state": session.current_state.value,
            "crisis_severity": session.crisis_severity.value,
            "message_count": len(session.conversation_history),
            "crisis_flags": session.crisis_flags,
            "state_age_minutes": (session.updated_at - session.created_at).total_seconds() / 60,
            "should_continue": self.should_continue_conversation(session_id)
        }


# Global state router instance
state_router = StateRouter()