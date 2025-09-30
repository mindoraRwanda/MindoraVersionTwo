"""
Crisis Intervention System for Mental Health Chatbot

This module provides high-priority crisis detection and intervention capabilities
that override normal conversation flow when users express suicidal ideation,
self-harm, or other mental health emergencies.
"""

import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass

from backend.app.services.session_state_manager import (
    SessionStateManager, ConversationState, CrisisSeverity, session_manager
)


class CrisisKeyword(Enum):
    """Categories of crisis keywords"""
    SUICIDE = "suicide"
    SELF_HARM = "self_harm"
    VIOLENCE = "violence"
    DESPAIR = "despair"
    HELPLESSNESS = "helplessness"
    ISOLATION = "isolation"


@dataclass
class CrisisPattern:
    """Pattern for detecting crisis indicators"""
    keywords: List[str]
    category: CrisisKeyword
    severity: CrisisSeverity
    regex_pattern: Optional[str] = None


class CrisisInterceptor:
    """High-priority crisis detection and intervention system"""

    def __init__(self):
        self.crisis_patterns = self._initialize_crisis_patterns()
        self.emergency_resources = self._initialize_emergency_resources()

    def _initialize_crisis_patterns(self) -> List[CrisisPattern]:
        """Initialize crisis detection patterns"""
        return [
            # Suicide indicators - Critical severity
            CrisisPattern(
                keywords=["kill myself", "end it all", "not worth living", "better off dead",
                         "suicide", "kill me", "end my life", "die by suicide"],
                category=CrisisKeyword.SUICIDE,
                severity=CrisisSeverity.CRITICAL,
                regex_pattern=r'\b(kill myself|end (it|my life|everything)|not worth living|better off dead)\b'
            ),

            # Self-harm indicators - High severity
            CrisisPattern(
                keywords=["cut myself", "hurt myself", "harm myself", "self harm",
                         "cutting", "burn myself", "overdose"],
                category=CrisisKeyword.SELF_HARM,
                severity=CrisisSeverity.HIGH,
                regex_pattern=r'\b(cut|hurt|harm|burn) myself|self.?harm|overdose\b'
            ),

            # Violence indicators - High severity
            CrisisPattern(
                keywords=["kill someone", "hurt people", "attack", "violent",
                         "murder", "shoot", "stab"],
                category=CrisisKeyword.VIOLENCE,
                severity=CrisisSeverity.HIGH,
                regex_pattern=r'\b(kill|hurt|attack|murder|shoot|stab) (someone|people|them|her|him)\b'
            ),

            # Despair indicators - Medium severity
            CrisisPattern(
                keywords=["hopeless", "worthless", "no point", "give up",
                         "can't go on", "everything is pointless"],
                category=CrisisKeyword.DESPAIR,
                severity=CrisisSeverity.MEDIUM,
                regex_pattern=r'\b(hopeless|worthless|no point|give up|can\'t go on|pointless)\b'
            ),

            # Helplessness indicators - Medium severity
            CrisisPattern(
                keywords=["can't cope", "falling apart", "losing control",
                         "everything too much", "can't handle"],
                category=CrisisKeyword.HELPLESSNESS,
                severity=CrisisSeverity.MEDIUM,
                regex_pattern=r'\b(can\'t (cope|handle|deal)|falling apart|losing control|too much)\b'
            ),

            # Isolation indicators - Low to Medium severity
            CrisisPattern(
                keywords=["alone", "no one cares", "nobody understands",
                         "isolated", "lonely", "no friends"],
                category=CrisisKeyword.ISOLATION,
                severity=CrisisSeverity.LOW,
                regex_pattern=r'\b(alone|no one cares|nobody understands|isolated|lonely)\b'
            ),
        ]

    def _initialize_emergency_resources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize emergency resources for different regions"""
        return {
            "rwanda": {
                "crisis_lines": [
                    "National Suicide Prevention Line: 114",
                    "Mental Health Support: +250 788 123 456",
                    "Emergency Services: 112"
                ],
                "organizations": [
                    "Rwanda Mental Health Association",
                    "BasicNeeds Rwanda",
                    "CARITAS Rwanda"
                ],
                "immediate_actions": [
                    "Contact emergency services if in immediate danger",
                    "Reach out to a trusted friend or family member",
                    "Go to your nearest health center"
                ]
            },
            "general": {
                "crisis_lines": [
                    "International Crisis Line: +44 20 8123 4567",
                    "Emergency Services: 911 (US), 999 (UK), 112 (EU)"
                ],
                "organizations": [
                    "International Association for Suicide Prevention",
                    "Befrienders Worldwide",
                    "Mental Health America"
                ],
                "immediate_actions": [
                    "Contact emergency services if in immediate danger",
                    "Call a crisis hotline immediately",
                    "Reach out to a trusted person"
                ]
            }
        }

    def detect_crisis(self, message: str, emotion_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str], CrisisSeverity]:
        """
        Detect crisis indicators in user message

        Returns:
            Tuple of (is_crisis, crisis_flags, highest_severity)
        """
        message_lower = message.lower()
        detected_flags = []
        highest_severity = CrisisSeverity.LOW

        # Check keyword patterns
        for pattern in self.crisis_patterns:
            # Check direct keyword matches
            for keyword in pattern.keywords:
                if keyword.lower() in message_lower:
                    detected_flags.append(f"{pattern.category.value}: {keyword}")
                    if pattern.severity.value > highest_severity.value:
                        highest_severity = pattern.severity
                    break

            # Check regex patterns if available
            if pattern.regex_pattern and not any(flag.startswith(pattern.category.value) for flag in detected_flags):
                if re.search(pattern.regex_pattern, message_lower, re.IGNORECASE):
                    detected_flags.append(f"{pattern.category.value}: regex_match")
                    if pattern.severity.value > highest_severity.value:
                        highest_severity = pattern.severity

        # Check emotion data for additional crisis indicators
        if emotion_data:
            crisis_score = self._analyze_emotion_crisis_indicators(emotion_data)
            if crisis_score >= 0.7:  # High crisis probability from emotion
                detected_flags.append("emotion_crisis: high_intensity")
                if highest_severity.value < CrisisSeverity.HIGH.value:
                    highest_severity = CrisisSeverity.HIGH

        is_crisis = len(detected_flags) > 0
        return is_crisis, detected_flags, highest_severity

    def _analyze_emotion_crisis_indicators(self, emotion_data: Dict[str, Any]) -> float:
        """Analyze emotion data for crisis indicators"""
        crisis_score = 0.0

        # High intensity negative emotions
        negative_emotions = ['fear', 'sadness', 'anger', 'disgust']
        for emotion in negative_emotions:
            if emotion in emotion_data:
                intensity = emotion_data[emotion]
                if isinstance(intensity, (int, float)) and intensity > 0.8:
                    crisis_score += 0.3

        # Very low positive emotions
        if 'joy' in emotion_data:
            joy_score = emotion_data['joy']
            if isinstance(joy_score, (int, float)) and joy_score < 0.2:
                crisis_score += 0.2

        # High distress indicators
        if 'distress' in emotion_data:
            distress = emotion_data['distress']
            if isinstance(distress, (int, float)) and distress > 0.7:
                crisis_score += 0.4

        return min(crisis_score, 1.0)

    def get_crisis_response(self, crisis_flags: List[str], severity: CrisisSeverity,
                          region: str = "rwanda") -> Dict[str, Any]:
        """Generate crisis intervention response"""
        response_data = {
            "is_crisis": True,
            "severity": severity.value,
            "crisis_flags": crisis_flags,
            "immediate_response": self._get_immediate_crisis_message(severity),
            "resources": self._get_regional_resources(region),
            "follow_up_actions": self._get_follow_up_actions(severity, region),
            "professional_help": self._get_professional_help_message(severity)
        }

        return response_data

    def _get_immediate_crisis_message(self, severity: CrisisSeverity) -> str:
        """Get immediate crisis intervention message"""
        if severity == CrisisSeverity.CRITICAL:
            return (
                "I hear that you're in a lot of pain right now, and I'm really concerned about you. "
                "Your life matters, and there are people who care about you and want to help. "
                "Please reach out to emergency services or a crisis hotline immediately."
            )
        elif severity == CrisisSeverity.HIGH:
            return (
                "I'm very worried about what you're going through. You don't have to face this alone. "
                "Please consider reaching out to a mental health professional or crisis service right away."
            )
        else:
            return (
                "I can sense that you're struggling right now. It's okay to ask for help, "
                "and there are people who want to support you through this difficult time."
            )

    def _get_regional_resources(self, region: str) -> Dict[str, Any]:
        """Get emergency resources for specific region"""
        return self.emergency_resources.get(region.lower(),
                                          self.emergency_resources["general"])

    def _get_follow_up_actions(self, severity: CrisisSeverity, region: str) -> List[str]:
        """Get follow-up actions based on severity"""
        base_actions = [
            "Take a moment to breathe and ground yourself",
            "Consider talking to someone you trust",
            "Remember that these feelings can pass with time and support"
        ]

        if severity in [CrisisSeverity.HIGH, CrisisSeverity.CRITICAL]:
            base_actions.insert(0, "Contact emergency services if you feel unsafe")
            base_actions.insert(1, "Call a crisis hotline immediately")

        return base_actions

    def _get_professional_help_message(self, severity: CrisisSeverity) -> str:
        """Get message encouraging professional help"""
        if severity == CrisisSeverity.CRITICAL:
            return (
                "This is an emergency situation. Please contact emergency services (112 in Rwanda) "
                "or go to your nearest hospital immediately. You are not alone in this."
            )
        elif severity == CrisisSeverity.HIGH:
            return (
                "I strongly recommend speaking with a mental health professional as soon as possible. "
                "They have the expertise to provide you with the specialized help you need."
            )
        else:
            return (
                "Consider reaching out to a counselor or therapist who can provide "
                "professional support tailored to your needs."
            )

    async def intercept_and_respond(self, session_id: str, message: str,
                                  emotion_data: Optional[Dict[str, Any]] = None,
                                  region: str = "rwanda") -> Optional[Dict[str, Any]]:
        """
        Main crisis interception method

        Returns crisis response if crisis detected, None otherwise
        """
        is_crisis, crisis_flags, severity = self.detect_crisis(message, emotion_data)

        if is_crisis:
            # Set crisis state in session
            session_manager.set_crisis_state(session_id, crisis_flags, severity)

            # Update session state to crisis intervention
            session_manager.update_session_state(
                session_id,
                ConversationState.CRISIS_INTERVENTION,
                {"crisis_flags": crisis_flags, "severity": severity.value}
            )

            # Generate crisis response
            crisis_response = self.get_crisis_response(crisis_flags, severity, region)

            # Add crisis response to session history
            session_manager.add_message_to_history(
                session_id,
                "assistant",
                crisis_response["immediate_response"],
                {"crisis_intervention": True, "severity": severity.value}
            )

            return crisis_response

        return None

    def should_override_conversation(self, session_id: str) -> bool:
        """Check if crisis state should override normal conversation"""
        session = session_manager.get_session(session_id)
        if not session:
            return False

        return (session.current_state == ConversationState.CRISIS_INTERVENTION or
                session.crisis_severity in [CrisisSeverity.HIGH, CrisisSeverity.CRITICAL])

    def clear_crisis_state(self, session_id: str) -> bool:
        """Clear crisis state and return to normal conversation"""
        session = session_manager.get_session(session_id)
        if not session:
            return False

        # Clear crisis flags
        session.crisis_flags.clear()
        session.crisis_severity = CrisisSeverity.LOW

        # Return to appropriate state based on conversation history
        if len(session.conversation_history) <= 2:
            session.current_state = ConversationState.INITIAL_DISTRESS
        else:
            session.current_state = ConversationState.GUIDING_IN_PROGRESS

        session.update_activity()
        return True


# Global crisis interceptor instance
crisis_interceptor = CrisisInterceptor()