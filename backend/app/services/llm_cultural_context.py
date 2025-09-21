"""
Cultural context and response generation for Rwanda-specific mental health support.
"""
from typing import Dict, List, Any
from .llm_config import (
    RWANDA_CULTURAL_CONTEXT, RWANDA_CRISIS_RESOURCES,
    EMOTION_RESPONSES, TOPIC_ADJUSTMENTS, GROUNDING_EXERCISE,
    SYSTEM_PROMPT_TEMPLATE
)


class RwandaCulturalManager:
    """Manages Rwanda-specific cultural context and resources."""

    @staticmethod
    def get_crisis_resources() -> Dict[str, Any]:
        """Get Rwanda-specific crisis resources"""
        return RWANDA_CRISIS_RESOURCES

    @staticmethod
    def get_cultural_context() -> Dict[str, str]:
        """Get Rwanda-specific cultural context"""
        return RWANDA_CULTURAL_CONTEXT

    @staticmethod
    def get_grounding_exercise() -> str:
        """Get Rwanda-culturally appropriate grounding exercise"""
        return GROUNDING_EXERCISE


class ResponseApproachManager:
    """Manages contextual response approaches based on emotion and topic."""

    @staticmethod
    def get_contextual_response_approach(
        emotion: str,
        user_message: str,
        conversation_context: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """Generate contextually appropriate response approach"""
        lowered = user_message.lower()
        cultural_elements = RWANDA_CULTURAL_CONTEXT

        # Default approach
        approach = {
            "tone": "empathetic",
            "cultural_element": "",
            "validation": "",
            "exploration_question": "",
            "support_offering": ""
        }

        # Emotion-specific approaches with cultural integration
        if emotion in EMOTION_RESPONSES:
            emotion_response = EMOTION_RESPONSES[emotion]
            approach.update({
                "tone": emotion_response["tone"],
                "cultural_element": cultural_elements.get("ubuntu_philosophy", ""),
                "validation": emotion_response["validation"],
                "exploration_question": emotion_response["exploration_question"],
                "support_offering": emotion_response["support_offering"]
            })
        else:
            # Default neutral response
            approach.update({
                "validation": "Thank you for sharing what's on your mind.",
                "exploration_question": "What would be most helpful for us to focus on today?",
                "support_offering": "I'm here to support you in whatever way feels right."
            })

        # Topic-specific adjustments
        for topic, adjustments in TOPIC_ADJUSTMENTS.items():
            if topic in lowered:
                if "cultural_element" in adjustments:
                    approach["cultural_element"] = cultural_elements.get(
                        adjustments["cultural_element"], ""
                    )
                if "exploration_question" in adjustments:
                    approach["exploration_question"] = adjustments["exploration_question"]
                break

        return approach

    @staticmethod
    def build_system_prompt(
        context_parts: List[str],
        emotion: str,
        response_approach: Dict[str, str]
    ) -> str:
        """Build contextual system prompt with Rwanda-specific context"""
        crisis_resources = RWANDA_CRISIS_RESOURCES

        context_str = "\n".join(context_parts) if context_parts else "This appears to be a new conversation."

        return SYSTEM_PROMPT_TEMPLATE.format(
            context=context_str,
            emotion=emotion,
            validation=response_approach.get('validation', ''),
            support_offering=response_approach.get('support_offering', ''),
            crisis_helpline=crisis_resources['national_helpline'],
            emergency=crisis_resources['emergency']
        )


class ConversationContextManager:
    """Manages conversation context and memory."""

    @staticmethod
    def build_memory_block(conversation_history: List[Dict[str, Any]], max_messages: int = 15) -> str:
        """Build memory block from conversation history"""
        if not conversation_history:
            return ""

        recent_messages = conversation_history[-max_messages:]
        return "\n".join(f"{m['role'].title()}: {m['text']}" for m in recent_messages)

    @staticmethod
    def is_simple_greeting(user_message: str) -> bool:
        """Check if message is a simple greeting"""
        from .llm_config import SIMPLE_GREETINGS
        return (
            len(user_message.strip()) < 15 and
            any(greeting in user_message.lower() for greeting in SIMPLE_GREETINGS)
        )

    @staticmethod
    def should_skip_analysis(user_message: str, skip_analysis: bool = False) -> bool:
        """Determine if expensive analysis should be skipped"""
        return skip_analysis or len(user_message.strip()) < 10