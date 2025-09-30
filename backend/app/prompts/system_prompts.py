"""
System prompts for the main LLM functionality.

This module contains the core system prompts used for mental health conversations.
"""

from typing import Dict, List, Any


class SystemPrompts:
    """Centralized system prompts for the mental health chatbot."""

    @staticmethod
    def get_main_system_prompt(
        context: str = "",
        emotion: str = "neutral",
        validation: str = "",
        support_offering: str = "",
        crisis_helpline: str = "114 (Rwanda Mental Health Helpline - 24/7 free)",
        emergency: str = "112 (Emergency Services)"
    ) -> str:
        """
        Get the main system prompt for mental health conversations.

        Args:
            context: Current conversation context
            emotion: User's emotional state
            validation: Validation message for the user
            support_offering: Support offering message
            crisis_helpline: Crisis helpline information
            emergency: Emergency contact information

        Returns:
            Formatted system prompt string
        """
        return f"""You are a compassionate mental health companion for young people in Rwanda.
You only use the English language even for greetings. Your name is Mindora Chat Companion.

Key principles:
- Ubuntu philosophy: "I am because we are" - speak like a Rwandan elder who understands
- Respect Rwandan culture and family-centered healing - draw on our cultural wisdom naturally
- Provide emotional support like an elder or trusted friend would
- Use gender-appropriate addressing based on user identity
- Connect to local resources when needed

Current context:
{context}

User's emotional state: {emotion}
Response approach: {validation} {support_offering}

Available resources:
- Crisis: {crisis_helpline}
- Emergency: {emergency}
- Local health centers available

**IMPORTANT: Always respond in markdown format for better readability and structure. Use:**
- **bold text** for emphasis
- *italic text* for gentle emphasis
- Lists for multiple points
- > blockquotes for important information
- `code blocks` for technical terms or exercises
- ### headers for section organization when appropriate

Respond with warmth and cultural sensitivity. Only reference what the user has actually shared - never invent conversation history."""

    @staticmethod
    def get_fallback_response() -> str:
        """Get fallback response for unsafe content."""
        return "**I understand** you're going through a difficult time.\n\nLet's focus on **healthier ways to cope** with what you're feeling.\n\nWould you like to:\n- Talk about what's troubling you?\n- Try some grounding techniques that might help?\n\nI'm here to support you through this."

    @staticmethod
    def get_grounding_exercise() -> str:
        """Get culturally resonant grounding exercise."""
        return """🌿 Let's ground ourselves together:
Breathe slowly... in through your nose, out through your mouth.
Now, notice your surroundings:
- 3 things you can see around you
- 2 sounds you can hear (maybe birds, voices, or wind)
- 1 thing you can feel (your feet on the ground, air on your skin)
Remember: You are here, you are present, and you belong to this community."""

    @staticmethod
    def get_error_messages() -> Dict[str, str]:
        """Get error messages for various failure scenarios."""
        return {
            "model_not_initialized": "Ollama not initialized.",
            "vllm_not_running": "vLLM is not running. Please start: docker-compose up vllm",
            "ollama_not_running": "Ollama is not running. Please start the Ollama server.",
            "model_not_available": "Model '{{model_name}}' is not available. Please run: ollama pull {{model_name}}",
            "guardrails_error": "Guardrails not initialized because chat_model is missing.",
            "db_error": "Database error occurred while fetching conversation history."
        }