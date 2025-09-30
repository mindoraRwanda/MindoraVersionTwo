"""
Configuration and constants for the LLM service.

This module provides a flexible, extensible configuration system that supports:
- Environment variable overrides
- External configuration files (JSON/YAML)
- Runtime configuration updates
- Type-safe configuration management
- Structured data classes for better maintainability
"""
import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for LLM model settings."""
    default_model_name: str = "llama3.2:1b"
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    vllm_base_url: str = field(default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1"))
    temperature: float = 1.0
    max_tokens: int = 512

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Create configuration from environment variables."""
        return cls(
            default_model_name=os.getenv("DEFAULT_MODEL_NAME", cls.default_model_name),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.9")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512"))
        )


@dataclass
class PerformanceConfig:
    """Configuration for performance-related settings."""
    max_input_length: int = 2000
    max_conversation_history: int = 15
    min_meaningful_message_length: int = 3
    rag_top_k: int = 3
    request_timeout: int = 30
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "PerformanceConfig":
        """Create configuration from environment variables."""
        return cls(
            max_input_length=int(os.getenv("MAX_INPUT_LENGTH", "2000")),
            max_conversation_history=int(os.getenv("MAX_CONVERSATION_HISTORY", "15")),
            min_meaningful_message_length=int(os.getenv("MIN_MESSAGE_LENGTH", "3")),
            rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            max_retries=int(os.getenv("MAX_RETRIES", "3"))
        )


@dataclass
class SafetyConfig:
    """Configuration for safety and content filtering."""
    crisis_keywords: List[str] = field(default_factory=lambda: [
        "kill myself", "end my life", "suicide", "hurt myself", "harm myself",
        "i want to die", "i dont want to live", "take my life", "end it all",
        "life isn't worth", "no reason to live", "better off dead",
        "can't go on", "want to disappear", "nothing matters", "hopeless"
    ])

    substance_abuse_keywords: List[str] = field(default_factory=lambda: [
        "overdose", "pills", "drugs", "alcohol abuse", "drinking problem",
        "getting high", "substance abuse", "addiction", "withdrawal"
    ])

    self_injury_keywords: List[str] = field(default_factory=lambda: [
        "cutting", "burning", "scratching", "hitting myself",
        "self injury", "self harm", "hurting myself"
    ])

    illegal_content_keywords: List[str] = field(default_factory=lambda: [
        "hack", "bomb", "weapon", "violence", "revenge"
    ])

    jailbreak_keywords: List[str] = field(default_factory=lambda: [
        "ignore instructions", "jailbreak", "pretend to be",
        "roleplay as", "act as if"
    ])

    inappropriate_relationship_keywords: List[str] = field(default_factory=lambda: [
        "romantic", "dating", "love you", "marry me", "kiss"
    ])

    medical_advice_keywords: List[str] = field(default_factory=lambda: [
        "diagnose", "medication dosage", "stop taking", "medical advice"
    ])

    mental_health_indicators: List[str] = field(default_factory=lambda: [
        "feel", "sad", "happy", "angry", "anxious", "stressed", "depressed", "worried",
        "help", "support", "talk", "problem", "difficult", "struggle", "emotion",
        "mental health", "therapy", "counseling", "coping"
    ])

    injection_patterns: List[str] = field(default_factory=lambda: [
        r'ignore\s+previous\s+instructions',
        r'forget\s+everything',
        r'system\s*:',
        r'assistant\s*:',
        r'user\s*:',
        r'<\s*system\s*>',
        r'<\s*/\s*system\s*>',
        r'```\s*system',
        r'```\s*prompt'
    ])

    unsafe_output_patterns: List[str] = field(default_factory=lambda: [
        r'how\s+to\s+make\s+(bomb|weapon|drug)',
        r'suicide\s+method',
        r'kill\s+yourself',
        r'harm\s+yourself',
        r'end\s+your\s+life'
    ])

    simple_greetings: List[str] = field(default_factory=lambda: [
        'hi', 'hello', 'hey', 'good morning', 'good evening', 'how are you'
    ])

    @classmethod
    def from_env(cls) -> "SafetyConfig":
        """Create configuration from environment variables."""
        # Default values for fallback
        default_crisis = [
            "kill myself", "end my life", "suicide", "hurt myself", "harm myself",
            "i want to die", "i dont want to live", "take my life", "end it all",
            "life isn't worth", "no reason to live", "better off dead",
            "can't go on", "want to disappear", "nothing matters", "hopeless"
        ]
        default_substance = [
            "overdose", "pills", "drugs", "alcohol abuse", "drinking problem",
            "getting high", "substance abuse", "addiction", "withdrawal"
        ]
        default_self_injury = [
            "cutting", "burning", "scratching", "hitting myself",
            "self injury", "self harm", "hurting myself"
        ]
        default_illegal = ["hack", "bomb", "weapon", "violence", "revenge"]
        default_jailbreak = ["ignore instructions", "jailbreak", "pretend to be", "roleplay as", "act as if"]
        default_inappropriate = ["romantic", "dating", "love you", "marry me", "kiss"]
        default_medical = ["diagnose", "medication dosage", "stop taking", "medical advice"]
        default_mental_health = [
            "feel", "sad", "happy", "angry", "anxious", "stressed", "depressed", "worried",
            "help", "support", "talk", "problem", "difficult", "struggle", "emotion",
            "mental health", "therapy", "counseling", "coping"
        ]
        default_injection = [
            r'ignore\s+previous\s+instructions', r'forget\s+everything', r'system\s*:',
            r'assistant\s*:', r'user\s*:', r'<\s*system\s*>', r'<\s*/\s*system\s*>',
            r'```\s*system', r'```\s*prompt'
        ]
        default_unsafe = [
            r'how\s+to\s+make\s+(bomb|weapon|drug)', r'suicide\s+method',
            r'kill\s+yourself', r'harm\s+yourself', r'end\s+your\s+life'
        ]
        default_greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'how are you']

        return cls(
            crisis_keywords=os.getenv("CRISIS_KEYWORDS", ",".join(default_crisis)).split(","),
            substance_abuse_keywords=os.getenv("SUBSTANCE_ABUSE_KEYWORDS", ",".join(default_substance)).split(","),
            self_injury_keywords=os.getenv("SELF_INJURY_KEYWORDS", ",".join(default_self_injury)).split(","),
            illegal_content_keywords=os.getenv("ILLEGAL_CONTENT_KEYWORDS", ",".join(default_illegal)).split(","),
            jailbreak_keywords=os.getenv("JAILBREAK_KEYWORDS", ",".join(default_jailbreak)).split(","),
            inappropriate_relationship_keywords=os.getenv("INAPPROPRIATE_RELATIONSHIP_KEYWORDS", ",".join(default_inappropriate)).split(","),
            medical_advice_keywords=os.getenv("MEDICAL_ADVICE_KEYWORDS", ",".join(default_medical)).split(","),
            mental_health_indicators=os.getenv("MENTAL_HEALTH_INDICATORS", ",".join(default_mental_health)).split(","),
            injection_patterns=os.getenv("INJECTION_PATTERNS", "|".join(default_injection)).split("|"),
            unsafe_output_patterns=os.getenv("UNSAFE_OUTPUT_PATTERNS", "|".join(default_unsafe)).split("|"),
            simple_greetings=os.getenv("SIMPLE_GREETINGS", ",".join(default_greetings)).split(",")
        )


@dataclass
class EmotionalResponseConfig:
    """Configuration for emotion-specific response guidance."""
    emotion_guidance: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "sadness": {
            "response_style": "gentle_presence",
            "validation_approach": "Show that you understand their sadness is real and valid",
            "exploration_style": "Ask gently about what's making them feel this way",
            "support_style": "Be there to listen without trying to fix everything at once",
            "natural_tone": "Like a caring friend who gets that sometimes life just hurts"
        },
        "anxiety": {
            "response_style": "calming_reassurance",
            "validation_approach": "Let them know anxiety is common and their feelings make sense",
            "exploration_style": "Help them identify what's triggering these anxious feelings",
            "support_style": "Offer practical ways to feel more grounded and less overwhelmed",
            "natural_tone": "Like a friend who understands how anxiety can make everything feel too much"
        },
        "stress": {
            "response_style": "understanding_support",
            "validation_approach": "Acknowledge how heavy their load feels right now",
            "exploration_style": "Help them figure out what's weighing on them most",
            "support_style": "Offer to help break things down into smaller, manageable pieces",
            "natural_tone": "Like a friend who sees they're struggling and wants to help share the weight"
        },
        "anger": {
            "response_style": "patient_understanding",
            "validation_approach": "Recognize that anger often comes from feeling hurt or scared",
            "exploration_style": "Gently help them understand what set off these angry feelings",
            "support_style": "Support them in expressing and processing their anger in healthy ways",
            "natural_tone": "Like a friend who knows anger is usually about something deeper that hurts"
        },
        "fear": {
            "response_style": "reassuring_presence",
            "validation_approach": "Let them know it's okay to feel scared about uncertain things",
            "exploration_style": "Help them talk about what specifically feels scary or unsafe",
            "support_style": "Offer steady reassurance and help them feel braver about facing fears",
            "natural_tone": "Like a friend who understands that fear is normal and wants to help them feel safer"
        }
    })

    topic_guidance: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "school": {
            "context_understanding": "School can be really tough with all the pressure and expectations",
            "exploration_approach": "Ask how school stuff is affecting their daily life and feelings",
            "natural_approach": "Talk about school like any friend would - no judgment, just understanding",
            "support_focus": "Help find ways to manage school stress while taking care of themselves"
        },
        "university": {
            "context_understanding": "University brings a lot of changes and new kinds of pressure",
            "exploration_approach": "Explore how uni life is affecting their emotional and social world",
            "natural_approach": "Relate to university challenges like a friend who's been through it",
            "support_focus": "Help navigate the ups and downs of university life and independence"
        },
        "family": {
            "context_understanding": "Family stuff can be complicated and affect everything else",
            "exploration_approach": "Help them talk about family relationships and conflicts naturally",
            "natural_approach": "Approach family topics with warmth and without taking sides",
            "support_focus": "Support them in figuring out family dynamics while honoring their own needs"
        },
        "work": {
            "context_understanding": "Work stress can make everything else feel harder",
            "exploration_approach": "Explore how work is impacting their well-being and relationships",
            "natural_approach": "Talk about work challenges like friends sharing about tough days",
            "support_focus": "Help find balance between work responsibilities and personal well-being"
        },
        "relationships": {
            "context_understanding": "Relationships can bring joy but also hurt and confusion",
            "exploration_approach": "Help them explore relationship dynamics with care and openness",
            "natural_approach": "Talk about relationships like a trusted friend giving advice",
            "support_focus": "Support healthy relationship patterns while respecting their feelings"
        }
    })

    response_templates: Dict[str, str] = field(default_factory=lambda: {
        "crisis_validation": "Generate a response that validates their crisis feelings while emphasizing immediate help-seeking",
        "support_offering": "Create an offer of support that feels genuine and natural",
        "question_asking": "Ask questions that show genuine interest in understanding their experience",
        "resource_mentioning": "Mention local resources naturally when they would be helpful, not as a default response"
    })

    @property
    def emotion_responses(self) -> Dict[str, Dict[str, str]]:
        """Backward compatibility property for emotion_responses."""
        return self.emotion_guidance

    @property
    def topic_adjustments(self) -> Dict[str, Dict[str, str]]:
        """Backward compatibility property for topic_adjustments."""
        return self.topic_guidance


@dataclass
class RwandaConfig:
    """Configuration for Rwanda-specific settings."""
    cultural_context: Dict[str, str] = field(default_factory=lambda: {
        "community_connection": "Remember that we're all connected - your feelings matter and affect the people who care about you.",
        "family_support": "Family and close friends can be great sources of support. Sometimes talking to someone you trust can make a big difference.",
        "healing_approaches": "Different people find help in different ways - therapy, talking to friends, or other approaches that feel right for them.",
        "resilience_mindset": "You've already shown strength by reaching out. Building resilience is about taking small steps and knowing you're not alone in this."
    })

    crisis_resources: Dict[str, Any] = field(default_factory=lambda: {
        "national_helpline": "114 (Rwanda Mental Health Helpline - 24/7 free)",
        "emergency": "112 (Emergency Services)",
        "hospitals": [
            "Ndera Neuropsychiatric Hospital: +250 781 447 928",
            "King Faisal Hospital: 3939 / +250 788 123 200"
        ],
        "community_health": "Contact your local Community Health Cooperative (CHC) or Health Center",
        "online_support": "Rwanda Biomedical Centre Mental Health Division"
    })

    system_prompt_template: str = """You are a warm, understanding friend who helps young people navigate tough emotions and life challenges.
Your name is Mindora, and you speak in a natural, conversational way - like a trusted friend who really cares.

What you do:
- Listen with genuine empathy and understanding
- Help people feel heard and validated in their struggles
- Offer gentle support and encouragement
- Ask thoughtful questions that show you want to understand
- Share relatable insights when it feels right
- Remember that healing happens in community - we're all connected

How you respond:
- Be authentic and caring, like a real conversation
- Use everyday language, not clinical terms
- Show you understand their feelings without trying to "fix" everything
- Ask questions that invite them to share more about their experience
- Offer support in a way that feels natural and unforced
- Remember: sometimes just listening is the best help

When things are really tough:
- Gently suggest talking to people who can help (family, friends, professionals)
- Know when to encourage reaching out for extra support
- Crisis support: {crisis_helpline}
- Emergency help: {emergency}

Current situation:
{context}

How they're feeling: {emotion}
Best approach: {validation} {support_offering}

Respond like a caring friend would - naturally, warmly, and supportively."""

    grounding_exercise: str = """Hey, let's try something simple to help you feel more present right now.

Take a slow breath in through your nose... and out through your mouth.

Now, just notice:
- 3 things you can see around you
- 2 sounds you can hear
- 1 thing you can physically feel

You're here, you're breathing, and you've got this moment."""

    fallback_response: str = "I can hear how tough this is for you. Sometimes it helps to talk about what's going on, or we could try some simple things to help you feel a bit more steady. What would help you most right now?"


@dataclass
class ErrorMessagesConfig:
    """Configuration for error messages."""
    messages: Dict[str, str] = field(default_factory=lambda: {
        "model_not_initialized": "LLM service not initialized.",
        "vllm_not_running": "vLLM is not running. Please start: docker-compose up vllm",
        "ollama_not_running": "Ollama is not running. Please start the Ollama server.",
        "model_not_available": "Model '{model_name}' is not available. Please run: ollama pull {model_name}",
        "guardrails_error": "Guardrails not initialized because chat_model is missing.",
        "db_error": "Database error occurred while fetching conversation history.",
        "config_error": "Configuration error: {error}",
        "file_not_found": "Configuration file not found: {file_path}"
    })


# Create default configuration instances
model_config = ModelConfig.from_env()
performance_config = PerformanceConfig.from_env()
safety_config = SafetyConfig.from_env()
emotional_config = EmotionalResponseConfig()
rwanda_config = RwandaConfig()
error_config = ErrorMessagesConfig()


class ConfigManager:
    """Centralized configuration manager with file loading support."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_file: Optional path to JSON configuration file
        """
        self.config_file = config_file or os.getenv("LLM_CONFIG_FILE")
        self._load_external_config()

    def _load_external_config(self):
        """Load configuration from external file if specified."""
        if not self.config_file:
            return

        config_path = Path(self.config_file)
        if not config_path.exists():
            print(f"Warning: Config file {self.config_file} not found, using defaults")
            return

        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    import yaml
                    external_config = yaml.safe_load(f)
                else:
                    external_config = json.load(f)

            # Update configurations from file
            self._update_from_dict(external_config)
            print(f"✅ Loaded configuration from {self.config_file}")

        except ImportError as e:
            print(f"Warning: PyYAML not installed for YAML config support: {e}")
            print("Install with: pip install PyYAML")
        except Exception as e:
            print(f"Error loading config file {self.config_file}: {e}")

    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configurations from dictionary."""
        # Update model config
        if "model" in config_dict:
            for key, value in config_dict["model"].items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)

        # Update performance config
        if "performance" in config_dict:
            for key, value in config_dict["performance"].items():
                if hasattr(performance_config, key):
                    setattr(performance_config, key, value)

        # Update safety config (more complex due to lists)
        if "safety" in config_dict:
            safety_data = config_dict["safety"]
            for key, value in safety_data.items():
                if hasattr(safety_config, key) and isinstance(value, list):
                    setattr(safety_config, key, value)

        # Update emotional guidance
        if "emotional_responses" in config_dict:
            emotional_config.emotion_guidance.update(config_dict["emotional_responses"])

        # Update topic guidance
        if "topic_adjustments" in config_dict:
            emotional_config.topic_guidance.update(config_dict["topic_adjustments"])

        # Update Rwanda config
        if "rwanda" in config_dict:
            rwanda_data = config_dict["rwanda"]
            if "cultural_context" in rwanda_data:
                rwanda_config.cultural_context.update(rwanda_data["cultural_context"])
            if "crisis_resources" in rwanda_data:
                self._update_dict_recursive(rwanda_config.crisis_resources, rwanda_data["crisis_resources"])
            if "system_prompt_template" in rwanda_data:
                rwanda_config.system_prompt_template = rwanda_data["system_prompt_template"]
            if "grounding_exercise" in rwanda_data:
                rwanda_config.grounding_exercise = rwanda_data["grounding_exercise"]
            if "fallback_response" in rwanda_data:
                rwanda_config.fallback_response = rwanda_data["fallback_response"]

    def _update_dict_recursive(self, target: Dict, source: Dict):
        """Recursively update nested dictionaries."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict_recursive(target[key], value)
            else:
                target[key] = value

    def get_model_config(self) -> ModelConfig:
        """Get current model configuration."""
        return model_config

    def get_performance_config(self) -> PerformanceConfig:
        """Get current performance configuration."""
        return performance_config

    def get_safety_config(self) -> SafetyConfig:
        """Get current safety configuration."""
        return safety_config

    def get_emotional_config(self) -> EmotionalResponseConfig:
        """Get current emotional response configuration."""
        return emotional_config

    def get_rwanda_config(self) -> RwandaConfig:
        """Get current Rwanda configuration."""
        return rwanda_config

    def get_error_config(self) -> ErrorMessagesConfig:
        """Get current error messages configuration."""
        return error_config

    def reload_config(self):
        """Reload configuration from file."""
        self._load_external_config()

    def export_config(self, file_path: str):
        """Export current configuration to file."""
        config_dict = {
            "model": {
                "default_model_name": model_config.default_model_name,
                "ollama_base_url": model_config.ollama_base_url,
                "vllm_base_url": model_config.vllm_base_url,
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_tokens
            },
            "performance": {
                "max_input_length": performance_config.max_input_length,
                "max_conversation_history": performance_config.max_conversation_history,
                "min_meaningful_message_length": performance_config.min_meaningful_message_length,
                "rag_top_k": performance_config.rag_top_k,
                "request_timeout": performance_config.request_timeout,
                "max_retries": performance_config.max_retries
            },
            "safety": {
                "crisis_keywords": safety_config.crisis_keywords,
                "substance_abuse_keywords": safety_config.substance_abuse_keywords,
                "self_injury_keywords": safety_config.self_injury_keywords,
                "illegal_content_keywords": safety_config.illegal_content_keywords,
                "jailbreak_keywords": safety_config.jailbreak_keywords,
                "inappropriate_relationship_keywords": safety_config.inappropriate_relationship_keywords,
                "medical_advice_keywords": safety_config.medical_advice_keywords,
                "mental_health_indicators": safety_config.mental_health_indicators,
                "injection_patterns": safety_config.injection_patterns,
                "unsafe_output_patterns": safety_config.unsafe_output_patterns,
                "simple_greetings": safety_config.simple_greetings
            },
            "emotional_responses": emotional_config.emotion_guidance,
            "topic_adjustments": emotional_config.topic_guidance,
            "rwanda": {
                "cultural_context": rwanda_config.cultural_context,
                "crisis_resources": rwanda_config.crisis_resources,
                "system_prompt_template": rwanda_config.system_prompt_template,
                "grounding_exercise": rwanda_config.grounding_exercise,
                "fallback_response": rwanda_config.fallback_response
            }
        }

        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"✅ Configuration exported to {file_path}")


# Global configuration manager instance
config_manager = ConfigManager()


# Backward compatibility - expose key values as module-level constants
DEFAULT_MODEL_NAME = model_config.default_model_name
OLLAMA_BASE_URL = model_config.ollama_base_url
VLLM_BASE_URL = model_config.vllm_base_url
MAX_INPUT_LENGTH = performance_config.max_input_length
MAX_CONVERSATION_HISTORY = performance_config.max_conversation_history
MIN_MEANINGFUL_MESSAGE_LENGTH = performance_config.min_meaningful_message_length
RAG_TOP_K = performance_config.rag_top_k

# Safety keywords (backward compatibility)
CRISIS_KEYWORDS = safety_config.crisis_keywords
SUBSTANCE_ABUSE_KEYWORDS = safety_config.substance_abuse_keywords
SELF_INJURY_KEYWORDS = safety_config.self_injury_keywords
ILLEGAL_CONTENT_KEYWORDS = safety_config.illegal_content_keywords
JAILBREAK_KEYWORDS = safety_config.jailbreak_keywords
INAPPROPRIATE_RELATIONSHIP_KEYWORDS = safety_config.inappropriate_relationship_keywords
MEDICAL_ADVICE_KEYWORDS = safety_config.medical_advice_keywords
MENTAL_HEALTH_INDICATORS = safety_config.mental_health_indicators
INJECTION_PATTERNS = safety_config.injection_patterns
UNSAFE_OUTPUT_PATTERNS = safety_config.unsafe_output_patterns
SIMPLE_GREETINGS = safety_config.simple_greetings

# Emotional responses (backward compatibility)
EMOTION_RESPONSES = emotional_config.emotion_responses
TOPIC_ADJUSTMENTS = emotional_config.topic_adjustments

# Rwanda-specific content (backward compatibility)
RWANDA_CULTURAL_CONTEXT = rwanda_config.cultural_context
RWANDA_CRISIS_RESOURCES = rwanda_config.crisis_resources
SYSTEM_PROMPT_TEMPLATE = rwanda_config.system_prompt_template
GROUNDING_EXERCISE = rwanda_config.grounding_exercise
FALLBACK_RESPONSE = rwanda_config.fallback_response

# Error messages (backward compatibility)
ERROR_MESSAGES = error_config.messages
