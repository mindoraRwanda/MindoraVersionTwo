"""
Configuration and constants for the LLM service.
"""
import os
from typing import Dict, List, Any


# Model Configuration
DEFAULT_MODEL_NAME = "gemma3:1b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")

# Performance Configuration
MAX_INPUT_LENGTH = 2000
MAX_CONVERSATION_HISTORY = 15
MIN_MEANINGFUL_MESSAGE_LENGTH = 3
RAG_TOP_K = 3

# Safety Configuration
CRISIS_KEYWORDS = [
    "kill myself", "end my life", "suicide", "hurt myself", "harm myself",
    "i want to die", "i dont want to live", "take my life", "end it all",
    "life isn't worth", "no reason to live", "better off dead",
    "can't go on", "want to disappear", "nothing matters", "hopeless"
]

SUBSTANCE_ABUSE_KEYWORDS = [
    "overdose", "pills", "drugs", "alcohol abuse", "drinking problem",
    "getting high", "substance abuse", "addiction", "withdrawal"
]

SELF_INJURY_KEYWORDS = [
    "cutting", "burning", "scratching", "hitting myself",
    "self injury", "self harm", "hurting myself"
]

ILLEGAL_CONTENT_KEYWORDS = [
    "hack", "bomb", "weapon", "violence", "revenge"
]

JAILBREAK_KEYWORDS = [
    "ignore instructions", "jailbreak", "pretend to be",
    "roleplay as", "act as if"
]

INAPPROPRIATE_RELATIONSHIP_KEYWORDS = [
    "romantic", "dating", "love you", "marry me", "kiss"
]

MEDICAL_ADVICE_KEYWORDS = [
    "diagnose", "medication dosage", "stop taking", "medical advice"
]

MENTAL_HEALTH_INDICATORS = [
    "feel", "sad", "happy", "angry", "anxious", "stressed", "depressed", "worried",
    "help", "support", "talk", "problem", "difficult", "struggle", "emotion",
    "mental health", "therapy", "counseling", "coping"
]

# Prompt Injection Patterns
INJECTION_PATTERNS = [
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

# Unsafe Output Patterns
UNSAFE_OUTPUT_PATTERNS = [
    r'how\s+to\s+make\s+(bomb|weapon|drug)',
    r'suicide\s+method',
    r'kill\s+yourself',
    r'harm\s+yourself',
    r'end\s+your\s+life'
]

# Simple Greetings for Fast Path
SIMPLE_GREETINGS = [
    'hi', 'hello', 'hey', 'good morning', 'good evening', 'how are you'
]

# Emotion-specific Response Templates
EMOTION_RESPONSES = {
    "sadness": {
        "tone": "gentle_presence",
        "validation": "Your sadness is valid, and it's brave of you to share these feelings.",
        "exploration_question": "What's weighing most heavily on your heart right now?",
        "support_offering": "I'm here to listen and walk through this with you."
    },
    "anxiety": {
        "tone": "calming_reassurance",
        "validation": "Anxiety can feel overwhelming, but you're not alone in this experience.",
        "exploration_question": "What thoughts or situations are contributing to this anxious feeling?",
        "support_offering": "Let's explore some grounding techniques that might help."
    },
    "stress": {
        "tone": "understanding_support",
        "validation": "It sounds like you're carrying a heavy load right now.",
        "exploration_question": "What aspect of this stress feels most urgent to address?",
        "support_offering": "We can break this down into manageable pieces together."
    }
}

# Topic-specific Adjustments
TOPIC_ADJUSTMENTS = {
    "school": {
        "exploration_question": "How are your studies affecting your overall well-being?"
    },
    "university": {
        "exploration_question": "How are your studies affecting your overall well-being?"
    },
    "family": {
        "cultural_element": "family_support",
        "exploration_question": "Family relationships can be complex. What's happening that you'd like to talk about?"
    },
    "work": {
        "exploration_question": "How is your work situation impacting your mental health?"
    },
    "job": {
        "exploration_question": "How is your work situation impacting your mental health?"
    }
}

# Rwanda-specific Cultural Context
RWANDA_CULTURAL_CONTEXT = {
    "ubuntu_philosophy": "Remember 'Ubuntu' - we are interconnected. Your pain affects the community, and the community is here to support you.",
    "family_support": "In Rwandan culture, family and community support are central to healing. Consider involving trusted family members or community leaders.",
    "traditional_healing": "While respecting traditional healing practices, professional mental health support can work alongside cultural approaches.",
    "resilience_history": "Rwanda has shown incredible resilience. Your personal healing contributes to our collective strength as a nation."
}

# Rwanda Crisis Resources
RWANDA_CRISIS_RESOURCES = {
    "national_helpline": "114 (Rwanda Mental Health Helpline - 24/7 free)",
    "emergency": "112 (Emergency Services)",
    "hospitals": [
        "Ndera Neuropsychiatric Hospital: +250 781 447 928",
        "King Faisal Hospital: 3939 / +250 788 123 200"
    ],
    "community_health": "Contact your local Community Health Cooperative (CHC) or Health Center",
    "online_support": "Rwanda Biomedical Centre Mental Health Division"
}

# System Prompt Template
SYSTEM_PROMPT_TEMPLATE = """You are a compassionate mental health companion for young people in Rwanda.
You only use the English language even for greetings. Your name is Mindora Chat Companion.

Key principles:
- Ubuntu philosophy: "I am because we are" - emphasize community support
- Respect Rwandan culture and family-centered healing
- Provide emotional support, not medical diagnosis
- Connect to local resources when needed

Current context:
{context}

User's emotional state: {emotion}
Response approach: {validation} {support_offering}

Available resources:
- Crisis: {crisis_helpline}
- Emergency: {emergency}
- Local health centers available

Respond with warmth and cultural sensitivity. Only reference what the user has actually shared - never invent conversation history."""

# Grounding Exercise Template
GROUNDING_EXERCISE = """🌿 Let's ground ourselves together:
Breathe slowly... in through your nose, out through your mouth.
Now, notice your surroundings:
- 3 things you can see around you
- 2 sounds you can hear (maybe birds, voices, or wind)
- 1 thing you can feel (your feet on the ground, air on your skin)
Remember: You are here, you are present, and you belong to this community."""

# Fallback Response for Unsafe Content
FALLBACK_RESPONSE = "I understand you're going through a difficult time. Let's focus on healthier ways to cope with what you're feeling. Would you like to talk about what's troubling you, or would you prefer some grounding techniques that might help?"

# Error Messages
ERROR_MESSAGES = {
    "model_not_initialized": "Ollama not initialized.",
    "vllm_not_running": "vLLM is not running. Please start: docker-compose up vllm",
    "ollama_not_running": "Ollama is not running. Please start the Ollama server.",
    "model_not_available": "Model '{model_name}' is not available. Please run: ollama pull {model_name}",
    "guardrails_error": "Guardrails not initialized because chat_model is missing.",
    "db_error": "Database error occurred while fetching conversation history."
}