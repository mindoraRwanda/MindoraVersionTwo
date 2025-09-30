"""
Externalized conversational content for the stateful dialogue system.

This module contains all conversational text decoupled from application logic,
organized by conversation states and available as structured content snippets.
"""
from typing import Dict, Any, Optional
from .llm_config import RWANDA_CULTURAL_CONTEXT, RWANDA_CRISIS_RESOURCES


class ConversationContentManager:
    """Manages all conversational content for the stateful dialogue system."""

    # Core conversational content organized by state and action
    CONTENT = {
        # Initial distress acknowledgment and probing
        "ack_and_probe": {
            "message": "I can hear that you're going through something difficult right now. Sometimes when we feel 'off,' it helps to understand a bit more about what's happening. Would you like to share a little more about how you're feeling?",
            "cultural_note": RWANDA_CULTURAL_CONTEXT["community_connection"]
        },

        # Elaboration acknowledgment
        "acknowledge_elaboration": {
            "message": "Thank you for sharing that with me. It sounds like you're dealing with something really challenging.",
            "transition": "I want to understand better so I can support you in the most helpful way."
        },

        # Clarifying questions based on user input
        "clarifying_questions": {
            "foggy": "When you say you feel 'foggy,' do you mean your mind feels cloudy and it's hard to think clearly, or more like an emotional haze where things just feel unclear?",
            "overwhelmed": "When you mention feeling overwhelmed, is it more about having too much on your plate, or a feeling that everything is just too intense emotionally?",
            "disconnected": "When you feel disconnected, do you mean from other people, from yourself, or from the world around you?",
            "default": "Could you help me understand a bit more about what this 'off' feeling is like for you? For example, is it more in your body, your thoughts, or your emotions?"
        },

        # Tool offerings (one at a time)
        "offer_grounding": {
            "introduction": "One approach that many people find helpful when feeling foggy or overwhelmed is a simple grounding exercise. This can help bring you back to the present moment.",
            "question": "Would you like to try a quick grounding exercise together?"
        },

        "offer_writing": {
            "introduction": "Another approach is to write down your thoughts and feelings. Sometimes getting them out of your head and onto paper can provide clarity and relief.",
            "question": "Would you be open to trying a brief writing exercise?"
        },

        "offer_breathing": {
            "introduction": "Deep breathing exercises can be really helpful for calming the mind and body when things feel foggy or overwhelming.",
            "question": "Would you like to try a simple breathing exercise?"
        },

        # Tool guidance content
        "grounding_intro": {
            "instruction": "Let's try a simple grounding exercise. This will help bring your attention back to the present moment.",
            "rwanda_context": RWANDA_CULTURAL_CONTEXT["resilience_mindset"]
        },

        "grounding_exercise": {
            "steps": """Take a slow breath in through your nose... and out through your mouth.

Now, just notice:
- 3 things you can see around you
- 2 sounds you can hear
- 1 thing you can physically feel

You're here, you're breathing, and you've got this moment.""",
            "follow_up": "How did that feel? Sometimes these simple exercises can help create a little space between us and our difficult feelings."
        },

        "writing_intro": {
            "instruction": "Let's try a brief writing exercise. Sometimes putting our thoughts into words can help us understand them better.",
            "prompt": "Take a moment to write about what's feeling 'off' right now. You don't need to share it with me unless you want to. Just write whatever comes to mind."
        },

        "breathing_intro": {
            "instruction": "Let's try a simple breathing exercise together. This can help calm the mind when things feel foggy.",
            "technique": "We'll do a 4-4-4 breathing pattern: breathe in for 4 counts, hold for 4, breathe out for 4."
        },

        # Alternative offerings
        "offer_alternative": {
            "acknowledgment": "That's completely okay - not every approach works for everyone.",
            "transition": "We could try something different instead."
        },

        # Conversation endings
        "gentle_close": {
            "message": "I want you to know that I'm here for you, and it's okay to reach out anytime. Take care of yourself.",
            "cultural_close": RWANDA_CULTURAL_CONTEXT["family_support"]
        },

        # Crisis intervention (high priority)
        "crisis_intervention": {
            "immediate_response": "I'm really concerned about what you're sharing, and I want you to know that you're not alone in this. Your life has value and there are people who can help right now.",
            "crisis_resources": f"Please reach out for immediate support:\n• Rwanda Mental Health Helpline: {RWANDA_CRISIS_RESOURCES['national_helpline']}\n• Emergency Services: {RWANDA_CRISIS_RESOURCES['emergency']}\n• Hospital Support: {RWANDA_CRISIS_RESOURCES['hospitals'][0]}",
            "cultural_support": RWANDA_CULTURAL_CONTEXT["community_connection"],
            "follow_up": "Would you like to talk about what's going on, or would you prefer to contact one of these resources right away?"
        }
    }

    # Crisis keywords for high-priority interrupt
    CRISIS_KEYWORDS = [
        "suicide", "kill myself", "end my life", "hurt myself", "harm myself",
        "i want to die", "i don't want to live", "take my life", "end it all",
        "life isn't worth", "no reason to live", "better off dead",
        "can't go on", "want to disappear", "nothing matters", "hopeless",
        "i'm done", "i give up", "i can't take this anymore"
    ]

    @classmethod
    def get_content(cls, content_key: str, sub_key: Optional[str] = None) -> str:
        """Retrieve conversational content by key."""
        content = cls.CONTENT.get(content_key, {})
        if sub_key:
            return content.get(sub_key, "")
        return content.get("message", "")

    @classmethod
    def get_crisis_keywords(cls) -> list:
        """Get list of crisis keywords for detection."""
        return cls.CRISIS_KEYWORDS

    @classmethod
    def format_crisis_response(cls, user_context: str = "") -> str:
        """Format crisis intervention response with user context."""
        base_response = cls.CONTENT["crisis_intervention"]

        if user_context:
            return f"{base_response['immediate_response']}\n\n{user_context}\n\n{base_response['crisis_resources']}\n\n{base_response['cultural_support']}\n\n{base_response['follow_up']}"
        else:
            return f"{base_response['immediate_response']}\n\n{base_response['crisis_resources']}\n\n{base_response['cultural_support']}\n\n{base_response['follow_up']}"


# Backward compatibility - expose content as module-level constants
CONVERSATION_CONTENT = ConversationContentManager.CONTENT
CRISIS_KEYWORDS = ConversationContentManager.CRISIS_KEYWORDS