from typing import Optional
from pydantic import field_validator
from pydantic_settings import SettingsConfigDict
from .base import BaseAppSettings

class ModelSettings(BaseAppSettings):
    """Configuration for LLM model settings."""

    # Core model configuration - maps to MODEL_NAME in env
    model_name: str = "gemma3:1b"  # Changed from default_model_name

    # Provider URLs - maps to OLLAMA_BASE_URL in env
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8001/v1"

    # Model parameters - maps to TEMPERATURE, MAX_TOKENS in env
    temperature: float = 0.6
    max_tokens: int = 500
    
    # LLM provider for general use (when using query engines)
    llm_provider: str = "groq"  # ollama, openai, groq, huggingface, etc.

    # API Keys (secrets) - maps to OPENAI_API_KEY, GROQ_API_KEY in env
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

    # LlamaIndex specific settings
    # Embedding model configuration
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_model_provider: str = "huggingface"  # huggingface, openai, etc.
    
    # LLM provider for LlamaIndex (when using query engines)
    llamaindex_llm_provider: str = "ollama"  # ollama, openai, groq, etc.
    llamaindex_model_name: str = "gemma3:1b"

    # Embedding API keys (if using OpenAI embeddings)
    openai_embedding_api_key: Optional[str] = None
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        # Align allowed temperature range with documentation and other components.
        if not 0.0 <= v <= 1.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
    
    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError('Max tokens must be positive')
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",  # Default env file
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        extra="allow"  # Allow extra fields from environment
        # No env_prefix so it reads MODEL_NAME, TEMPERATURE, etc. directly
    )