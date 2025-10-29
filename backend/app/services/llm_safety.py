# backend/app/services/safety_manager.py
import re
from typing import Optional
from backend.app.settings.settings import settings

class SafetyManager:
    """Lightweight input/output guardrails using project settings."""

    @staticmethod
    def sanitize_input(user_message: str) -> str:
        """Trim, redact simple injection patterns, and cap length."""
        sanitized = re.sub(r"\s+", " ", (user_message or "").strip())

        if settings and getattr(settings, "safety", None):
            for pattern in settings.safety.injection_patterns:
                try:
                    sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)
                except re.error:
                    # Skip invalid regexes rather than failing the request
                    continue

            max_len = getattr(getattr(settings, "performance", None), "max_input_length", 2000)
            if len(sanitized) > max_len:
                sanitized = sanitized[:max_len] + "..."

        return sanitized

    @staticmethod
    def is_safe_output(response: str) -> bool:
        """Return False if model output matches any unsafe pattern."""
        if not (settings and getattr(settings, "safety", None)):
            return True

        for pattern in settings.safety.unsafe_output_patterns:
            try:
                if re.search(pattern, response or "", re.IGNORECASE):
                    return False
            except re.error:
                continue
        return True

    @staticmethod
    def classify_intent(user_message: str) -> str:
        """Heuristic, config-driven intent classification."""
        text = (user_message or "").lower().strip()
        if not (settings and getattr(settings, "safety", None)):
            return "general"

        s = settings.safety
        # Crisis / safety buckets
        for kw in s.crisis_keywords:
            if kw in text:
                return "self_harm"
        for kw in s.substance_abuse_keywords:
            if kw in text:
                return "substance_abuse"
        for kw in s.self_injury_keywords:
            if kw in text:
                return "self_injury"
        if any(k in text for k in s.illegal_content_keywords):
            return "illegal"
        if any(k in text for k in s.jailbreak_keywords):
            return "jailbreak"
        if any(k in text for k in s.inappropriate_relationship_keywords):
            return "inappropriate_relationship"
        if any(k in text for k in s.medical_advice_keywords):
            return "medical_advice"

        if len(text) > 30 and not any(ind in text for ind in s.mental_health_indicators):
            return "off_topic"

        return "general"

    @staticmethod
    def get_safety_response(intent: str) -> Optional[str]:
        """Return a culturally-aware response for concerning intents."""
        if not (settings and getattr(settings, "cultural", None)):
            return None

        crisis = settings.cultural.crisis_resources
        culture = settings.cultural.cultural_context

        responses = {
            "self_harm": (
                "I'm deeply concerned about what you're sharing. Your life has value and you don't have to face this alone. "
                f"Please reach out for immediate help: {crisis['national_helpline']}, Emergency: {crisis['emergency']}, "
                f"or visit {crisis['hospitals'][0]}. {culture['ubuntu_philosophy']} I'm here to support you."
            ),
            "substance_abuse": (
                "I hear that you're dealing with substance-related challenges. This needs professional support. "
                f"Please contact {crisis['national_helpline']} or your nearest health center. {culture['family_support']}"
            ),
            "self_injury": (
                "I'm concerned about the self-harm you're describing. Hurting yourself isn’t the solution. "
                f"Please reach out: {crisis['national_helpline']} or {crisis['emergency']}. {culture['traditional_healing']}"
            ),
            "illegal": "I can't help with that. I'm here only to support mental health and well-being in a safe, constructive way.",
            "jailbreak": "I can't override my safety instructions. Let's focus on supporting your well-being today.",
            "inappropriate_relationship": (
                "I'm a supportive mental health companion, not for romantic relationships. "
                "Let's keep our conversation focused on your well-being."
            ),
            "medical_advice": (
                "I can't provide medical diagnosis or medication advice. Please consult qualified care at "
                f"{crisis['hospitals'][0]}, {crisis['hospitals'][1]}, or your local health center. "
                f"{crisis['community_health']}"
            ),
            "off_topic": (
                "I'm designed to support emotional well-being and mental health. "
                "Is there anything related to your mental health I can help you with today?"
            ),
        }
        return responses.get(intent)

    @staticmethod
    def check_safety(message: str) -> Optional[str]:
        """Convenience: classify and return a safety response if needed."""
        intent = SafetyManager.classify_intent(message)
        if intent in {
            "self_harm", "substance_abuse", "self_injury",
            "illegal", "jailbreak", "inappropriate_relationship",
            "medical_advice", "off_topic"
        }:
            return SafetyManager.get_safety_response(intent)
        return None
