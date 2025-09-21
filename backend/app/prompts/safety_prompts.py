"""
Safety prompts and guardrails for the mental health chatbot.

This module contains prompts and configurations for ensuring safe conversations.
"""

from typing import Dict, List, Any


class SafetyPrompts:
    """Centralized safety prompts and guardrails."""

    @staticmethod
    def get_guardrails_config() -> str:
        """
        Get the guardrails configuration for the chatbot.

        Returns:
            Colang configuration string for guardrails
        """
        return """colang: |
          flow safety_and_boundary_check
            user ".*"
            call classify_intent_with_llm $last_user_message

            if $result == "self_harm":
              bot "I'm deeply concerned about what you're sharing. Your life has value and you don't have to face this alone. Please reach out for immediate professional help: {crisis_helpline}, Emergency: {emergency}, or visit {hospital}. {ubuntu_philosophy} I'm here to support you, but professional crisis support is essential right now."
              stop
            elif $result == "substance_abuse":
              bot "I understand you're dealing with substance-related concerns. This is a serious matter that needs professional support. Please contact: {crisis_helpline} for mental health and addiction support, or visit your nearest health center. {family_support} Professional treatment combined with community support can make a real difference."
              stop
            elif $result == "self_injury":
              bot "I'm concerned about the self-harm you're describing. These feelings are valid, but hurting yourself isn't the solution. Please reach out: {crisis_helpline} or {emergency} for immediate support. {traditional_healing} Let's work together on healthier ways to manage these difficult emotions."
              stop
            elif $result == "illegal":
              bot "I can't help with that. I'm here only to support mental health and well-being in a safe, constructive way."
              stop
            elif $result == "jailbreak":
              bot "I can't override my safety instructions. I'm here only to support mental health and your well-being. Let's focus on how I can help you today."
              stop
            elif $result == "inappropriate_relationship":
              bot "I'm here as a supportive mental health companion, not for romantic or personal relationships. Let's keep our conversation focused on supporting your well-being and mental health. How can I help you today?"
              stop
            elif $result == "medical_advice":
              bot "I can't provide medical diagnosis or medication advice. For medical concerns, please consult qualified healthcare providers at: {hospital1}, {hospital2}, or your local health center. {community_health} I can offer emotional support and coping strategies while you seek professional medical care."
              stop
            elif $result == "off_topic":
              bot "I'm a mental health companion designed specifically to support your emotional well-being and mental health. While I'd love to help with other topics, my expertise is in providing mental health support, coping strategies, and connecting you with resources in Rwanda. Is there anything related to your mental health or well-being I can help you with today?"
              stop"""

    @staticmethod
    def get_crisis_keywords() -> List[str]:
        """Get keywords that indicate crisis situations."""
        return [
            "kill myself", "end my life", "suicide", "hurt myself", "harm myself",
            "i want to die", "i dont want to live", "take my life", "end it all",
            "life isn't worth", "no reason to live", "better off dead",
            "can't go on", "want to disappear", "nothing matters", "hopeless"
        ]

    @staticmethod
    def get_substance_abuse_keywords() -> List[str]:
        """Get keywords that indicate substance abuse concerns."""
        return [
            "overdose", "pills", "drugs", "alcohol abuse", "drinking problem",
            "getting high", "substance abuse", "addiction", "withdrawal"
        ]

    @staticmethod
    def get_self_injury_keywords() -> List[str]:
        """Get keywords that indicate self-injury concerns."""
        return [
            "cutting", "burning", "scratching", "hitting myself",
            "self injury", "self harm", "hurting myself"
        ]

    @staticmethod
    def get_illegal_content_keywords() -> List[str]:
        """Get keywords that indicate illegal content requests."""
        return [
            "hack", "bomb", "weapon", "violence", "revenge"
        ]

    @staticmethod
    def get_jailbreak_keywords() -> List[str]:
        """Get keywords that indicate jailbreak attempts."""
        return [
            "ignore instructions", "jailbreak", "pretend to be",
            "roleplay as", "act as if"
        ]

    @staticmethod
    def get_inappropriate_relationship_keywords() -> List[str]:
        """Get keywords that indicate inappropriate relationship requests."""
        return [
            "romantic", "dating", "love you", "marry me", "kiss"
        ]

    @staticmethod
    def get_medical_advice_keywords() -> List[str]:
        """Get keywords that indicate medical advice requests."""
        return [
            "diagnose", "medication dosage", "stop taking", "medical advice"
        ]

    @staticmethod
    def get_mental_health_indicators() -> List[str]:
        """Get keywords that indicate mental health concerns."""
        return [
            "feel", "sad", "happy", "angry", "anxious", "stressed", "depressed", "worried",
            "help", "support", "talk", "problem", "difficult", "struggle", "emotion",
            "mental health", "therapy", "counseling", "coping"
        ]

    @staticmethod
    def get_injection_patterns() -> List[str]:
        """Get regex patterns for prompt injection detection."""
        return [
            r'ignore\s+previous\s+instructions',
            r'forget\s+everything',
            r'system\s*:',
            r'assistant\s*:',
            r'user\s*:',
            r'<\s*system\s*>',
            r'<\s*/\s*system\s*>',
            r'```\s*system',
            r'```\s*prompt'
        ]

    @staticmethod
    def get_unsafe_output_patterns() -> List[str]:
        """Get regex patterns for unsafe output detection."""
        return [
            r'how\s+to\s+make\s+(bomb|weapon|drug)',
            r'suicide\s+method',
            r'kill\s+yourself',
            r'harm\s+yourself',
            r'end\s+your\s+life'
        ]

    @staticmethod
    def get_simple_greetings() -> List[str]:
        """Get simple greetings for fast path processing."""
        return [
            'hi', 'hello', 'hey', 'good morning', 'good evening', 'how are you'
        ]