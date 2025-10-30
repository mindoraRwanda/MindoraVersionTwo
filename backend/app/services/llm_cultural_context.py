from typing import Dict, List, Any
import random
# Use the compatibility layer for gradual migration
from ..settings.settings import settings
from ..prompts.system_prompts import SystemPrompts


class RwandaCulturalManager:
    """Manages Rwanda-specific cultural context and resources."""

    @staticmethod
    def get_crisis_resources(language: str = 'en') -> Dict[str, Any]:
        """Get Rwanda-specific crisis resources in the specified language"""
        from ..prompts.cultural_context_prompts import CulturalContextPrompts
        return CulturalContextPrompts.get_rwanda_crisis_resources(language)

    @staticmethod
    def get_cultural_context(language: str = 'en') -> Dict[str, str]:
        """Get Rwanda-specific cultural context in the specified language"""
        from ..prompts.cultural_context_prompts import CulturalContextPrompts
        contexts = CulturalContextPrompts.get_rwanda_cultural_context(language)
        # Return a random phrase for each context type
        result = {}
        for key, phrases in contexts.items():
            if isinstance(phrases, list) and phrases:
                result[key] = random.choice(phrases)
            else:
                result[key] = phrases if isinstance(phrases, str) else ""
        return result

    @staticmethod
    def get_grounding_exercise() -> str:
        """Get culturally resonant grounding exercise"""
        return SystemPrompts.get_grounding_exercise()


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
        cultural_elements = settings.cultural.cultural_context if settings.cultural else {}

        # Default approach
        approach = {
            "tone": "empathetic",
            "cultural_element": "",
            "validation": "",
            "exploration_question": "",
            "support_offering": ""
        }

        # Emotion-specific approaches with cultural integration
        emotion_responses = settings.emotional.emotion_responses if settings.emotional else {}
        
        if emotion in emotion_responses:
            emotion_response = emotion_responses[emotion]
            approach.update({
                "tone": emotion_response.get("natural_tone", "empathetic"),
                "cultural_element": cultural_elements.get("ubuntu_philosophy", ""),
                "validation": emotion_response.get("validation_approach", ""),
                "exploration_question": emotion_response.get("exploration_style", ""),
                "support_offering": emotion_response.get("support_style", "")
            })
        else:
            # Default neutral response
            approach.update({
                "validation": "Thank you for sharing what's on your mind.",
                "exploration_question": "What would be most helpful for us to focus on today?",
                "support_offering": "I'm here to support you in whatever way feels right."
            })

        # Topic-specific adjustments
        topic_adjustments = settings.emotional.topic_adjustments if settings.emotional else {}
        for topic, adjustments in topic_adjustments.items():
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
        response_approach: Dict[str, str],
        language: str = 'en'
    ) -> str:
        """Build contextual system prompt with Rwanda-specific context in the specified language"""
        crisis_resources = RwandaCulturalManager.get_crisis_resources(language)

        context_str = "\n".join(context_parts) if context_parts else "This appears to be a new conversation."

        if settings.cultural:
            return SystemPrompts.get_main_system_prompt(
                context=context_str,
                emotion=emotion,
                validation=response_approach.get('validation', ''),
                support_offering=response_approach.get('support_offering', ''),
                crisis_helpline=crisis_resources.get('national_helpline', ''),
                emergency=crisis_resources.get('emergency', ''),
                language=language
            )
        else:
            # Fallback basic prompt
            return f"You are a helpful mental health assistant. Context: {context_str}, Emotion: {emotion}"


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
        simple_greetings = settings.safety.simple_greetings if settings.safety else []
        return (
            len(user_message.strip()) < 15 and
            any(greeting in user_message.lower() for greeting in simple_greetings)
        )

    @staticmethod
    def should_skip_analysis(user_message: str, skip_analysis: bool = False) -> bool:
        """Determine if expensive analysis should be skipped"""
        return skip_analysis or len(user_message.strip()) < 10