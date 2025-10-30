import re
from typing import Dict, List, Any, Optional
from ..settings.settings import settings


class SafetyManager:
    """Manages safety checks and content filtering."""

    @staticmethod
    def sanitize_input(user_message: str) -> str:
        """Sanitize user input to prevent injection attacks and clean malicious content"""
        # Remove excessive whitespace and normalize
        sanitized = re.sub(r'\s+', ' ', user_message.strip())

        # Remove potential prompt injection patterns
        if settings.safety:
            for pattern in settings.safety.injection_patterns:
                sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)

        # Limit length to prevent context overflow
        max_length = settings.performance.max_input_length if settings.performance else 2000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."

        return sanitized

    @staticmethod
    def is_safe_output(response: str) -> bool:
        """Check if model output contains unsafe content"""
        if settings.safety:
            for pattern in settings.safety.unsafe_output_patterns:
                try:
                    if re.search(pattern, response, re.IGNORECASE):
                        return False
                except re.error as e:
                    print(f"Invalid regex pattern '{pattern}': {e}")
                    # Skip invalid patterns instead of crashing
                    continue
        return True

    @staticmethod
    def classify_intent(user_message: str) -> str:
        """Classify user intent for safety checking"""
        lowered = user_message.lower().strip()

        # Get safety configuration
        if not settings.safety:
            return "general"

        # Check for crisis situations
        for kw in settings.safety.crisis_keywords:
            if kw in lowered:
                return "self_harm"

        for kw in settings.safety.substance_abuse_keywords:
            if kw in lowered:
                return "substance_abuse"

        for kw in settings.safety.self_injury_keywords:
            if kw in lowered:
                return "self_injury"

        # Enhanced illegal content detection
        if any(k in lowered for k in settings.safety.illegal_content_keywords):
            return "illegal"

        # Enhanced jailbreak detection
        if any(k in lowered for k in settings.safety.jailbreak_keywords):
            return "jailbreak"

        # Inappropriate relationship boundaries
        if any(k in lowered for k in settings.safety.inappropriate_relationship_keywords):
            return "inappropriate_relationship"

        # Medical advice seeking
        if any(k in lowered for k in settings.safety.medical_advice_keywords):
            return "medical_advice"

        # Off-topic requests (non-mental health) - simple boundary check
        if len(lowered) > 30 and not any(indicator in lowered for indicator in settings.safety.mental_health_indicators):
            return "off_topic"

        return "general"

    @staticmethod
    def get_safety_response(intent: str) -> Optional[str]:
        """Generate appropriate safety response based on classified intent."""
        if not settings.cultural:
            return None
            
        crisis_resources = settings.cultural.crisis_resources
        cultural_context = settings.cultural.cultural_context
        
        safety_responses = {
            "self_harm": f"I'm deeply concerned about what you're sharing. Your life has value and you don't have to face this alone. Please reach out for immediate professional help: {crisis_resources['national_helpline']}, Emergency: {crisis_resources['emergency']}, or visit {crisis_resources['hospitals'][0]}. {cultural_context['ubuntu_philosophy']} I'm here to support you, but professional crisis support is essential right now.",
            
            "substance_abuse": f"I understand you're dealing with substance-related concerns. This is a serious matter that needs professional support. Please contact: {crisis_resources['national_helpline']} for mental health and addiction support, or visit your nearest health center. {cultural_context['family_support']} Professional treatment combined with community support can make a real difference.",
            
            "self_injury": f"I'm concerned about the self-harm you're describing. These feelings are valid, but hurting yourself isn't the solution. Please reach out: {crisis_resources['national_helpline']} or {crisis_resources['emergency']} for immediate support. {cultural_context['traditional_healing']} Let's work together on healthier ways to manage these difficult emotions.",
            
            "illegal": "I can't help with that. I'm here only to support mental health and well-being in a safe, constructive way.",
            
            "jailbreak": "I can't override my safety instructions. I'm here only to support mental health and your well-being. Let's focus on how I can help you today.",
            
            "inappropriate_relationship": "I'm here as a supportive mental health companion, not for romantic or personal relationships. Let's keep our conversation focused on supporting your well-being and mental health. How can I help you today?",
            
            "medical_advice": f"I can't provide medical diagnosis or medication advice. For medical concerns, please consult qualified healthcare providers at: {crisis_resources['hospitals'][0]}, {crisis_resources['hospitals'][1]}, or your local health center. {crisis_resources['community_health']} I can offer emotional support and coping strategies while you seek professional medical care.",
            
            "off_topic": "I'm a mental health companion designed specifically to support your emotional well-being and mental health. While I'd love to help with other topics, my expertise is in providing mental health support, coping strategies, and connecting you with resources in Rwanda. Is there anything related to your mental health or well-being I can help you with today?"
        }
        
        return safety_responses.get(intent)

    @staticmethod
    def check_safety(message: str) -> Optional[str]:
        """Check message safety and return appropriate response if needed."""
        intent = SafetyManager.classify_intent(message)
        
        # Return safety response for concerning intents
        if intent in ["self_harm", "substance_abuse", "self_injury", "illegal", "jailbreak", "inappropriate_relationship", "medical_advice", "off_topic"]:
            return SafetyManager.get_safety_response(intent)
            
        return None