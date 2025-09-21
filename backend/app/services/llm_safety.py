"""
Safety and guardrails functionality for the LLM service.
"""
import re
from typing import Dict, List, Any, Optional
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from .llm_config import (
    CRISIS_KEYWORDS, SUBSTANCE_ABUSE_KEYWORDS, SELF_INJURY_KEYWORDS,
    ILLEGAL_CONTENT_KEYWORDS, JAILBREAK_KEYWORDS, INAPPROPRIATE_RELATIONSHIP_KEYWORDS,
    MEDICAL_ADVICE_KEYWORDS, MENTAL_HEALTH_INDICATORS, INJECTION_PATTERNS,
    UNSAFE_OUTPUT_PATTERNS, RWANDA_CRISIS_RESOURCES, RWANDA_CULTURAL_CONTEXT,
    ERROR_MESSAGES
)


class SafetyManager:
    """Manages safety checks and content filtering."""

    @staticmethod
    def sanitize_input(user_message: str) -> str:
        """Sanitize user input to prevent injection attacks and clean malicious content"""
        # Remove excessive whitespace and normalize
        sanitized = re.sub(r'\s+', ' ', user_message.strip())

        # Remove potential prompt injection patterns
        for pattern in INJECTION_PATTERNS:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)

        # Limit length to prevent context overflow
        if len(sanitized) > 2000:
            sanitized = sanitized[:2000] + "..."

        return sanitized

    @staticmethod
    def is_safe_output(response: str) -> bool:
        """Check if model output contains unsafe content"""
        for pattern in UNSAFE_OUTPUT_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                return False
        return True

    @staticmethod
    def classify_intent(user_message: str) -> str:
        """Classify user intent for safety guardrails"""
        lowered = user_message.lower().strip()

        # Check for crisis situations
        for kw in CRISIS_KEYWORDS:
            if kw in lowered:
                return "self_harm"

        for kw in SUBSTANCE_ABUSE_KEYWORDS:
            if kw in lowered:
                return "substance_abuse"

        for kw in SELF_INJURY_KEYWORDS:
            if kw in lowered:
                return "self_injury"

        # Enhanced illegal content detection
        if any(k in lowered for k in ILLEGAL_CONTENT_KEYWORDS):
            return "illegal"

        # Enhanced jailbreak detection
        if any(k in lowered for k in JAILBREAK_KEYWORDS):
            return "jailbreak"

        # Inappropriate relationship boundaries
        if any(k in lowered for k in INAPPROPRIATE_RELATIONSHIP_KEYWORDS):
            return "inappropriate_relationship"

        # Medical advice seeking
        if any(k in lowered for k in MEDICAL_ADVICE_KEYWORDS):
            return "medical_advice"

        # Off-topic requests (non-mental health) - simple boundary check
        if len(lowered) > 30 and not any(indicator in lowered for indicator in MENTAL_HEALTH_INDICATORS):
            return "off_topic"

        return "general"

    @staticmethod
    def get_guardrails_config() -> str:
        """Generate guardrails configuration with Rwanda-specific responses"""
        crisis_resources = RWANDA_CRISIS_RESOURCES
        cultural_context = RWANDA_CULTURAL_CONTEXT

        return f"""
        colang: |
          flow safety_and_boundary_check
            user ".*"
            call classify_intent $last_user_message

            if $result == "self_harm":
              bot "I'm deeply concerned about what you're sharing. Your life has value and you don't have to face this alone. Please reach out for immediate professional help: {crisis_resources['national_helpline']}, Emergency: {crisis_resources['emergency']}, or visit {crisis_resources['hospitals'][0]}. {cultural_context['ubuntu_philosophy']} I'm here to support you, but professional crisis support is essential right now."
              stop
            elif $result == "substance_abuse":
              bot "I understand you're dealing with substance-related concerns. This is a serious matter that needs professional support. Please contact: {crisis_resources['national_helpline']} for mental health and addiction support, or visit your nearest health center. {cultural_context['family_support']} Professional treatment combined with community support can make a real difference."
              stop
            elif $result == "self_injury":
              bot "I'm concerned about the self-harm you're describing. These feelings are valid, but hurting yourself isn't the solution. Please reach out: {crisis_resources['national_helpline']} or {crisis_resources['emergency']} for immediate support. {cultural_context['traditional_healing']} Let's work together on healthier ways to manage these difficult emotions."
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
              bot "I can't provide medical diagnosis or medication advice. For medical concerns, please consult qualified healthcare providers at: {crisis_resources['hospitals'][0]}, {crisis_resources['hospitals'][1]}, or your local health center. {crisis_resources['community_health']} I can offer emotional support and coping strategies while you seek professional medical care."
              stop
            elif $result == "off_topic":
              bot "I'm a mental health companion designed specifically to support your emotional well-being and mental health. While I'd love to help with other topics, my expertise is in providing mental health support, coping strategies, and connecting you with resources in Rwanda. Is there anything related to your mental health or well-being I can help you with today?"
              stop
        """


class GuardrailsManager:
    """Manages NeMo guardrails initialization and execution."""

    def __init__(self, chat_model):
        self.chat_model = chat_model
        self.rails = None
        self._initialize_guardrails()

    def _initialize_guardrails(self):
        """Initialize guardrails with Rwanda-specific configuration"""
        try:
            if self.chat_model:
                config = RailsConfig.from_content(SafetyManager.get_guardrails_config())
                self.rails = LLMRails(config, llm=self.chat_model)
                print("Guardrails loaded successfully.")
            else:
                print(ERROR_MESSAGES["guardrails_error"])
        except Exception as e:
            print(f"Error loading guardrails: {e}")
            self.rails = None

    async def check_guardrails(self, message: str) -> Optional[str]:
        """Check message against guardrails and return safety response if needed"""
        if not self.rails:
            return None

        try:
            response = await self.rails.generate_async(messages=[{"role": "user", "content": message}])

            # If guardrails blocked the message, return the safety response
            if response and hasattr(response, 'content') and response.content:
                return response.content.strip()
        except Exception as e:
            print(f"Guardrails Error: {e}")
            # Continue with normal processing if guardrails fail

        return None


# Guardrails action for intent classification
@action()
async def classify_intent_with_ollama(last_user_message: str):
    """Action for NeMo guardrails to classify user intent"""
    return SafetyManager.classify_intent(last_user_message)