from pydantic import field_validator
from pydantic_settings import SettingsConfigDict
from .base import BaseAppSettings

class PerformanceSettings(BaseAppSettings):
    """Configuration for performance-related settings."""
    
    max_input_length: int = 2000
    max_conversation_history: int = 15
    min_meaningful_message_length: int = 3
    rag_top_k: int = 3
    request_timeout: int = 30
    max_retries: int = 3
    
    # Vector database settings
    vector_db_path: str = "./mental_health_knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    @field_validator('max_input_length', 'max_conversation_history', 'min_meaningful_message_length',
                     'rag_top_k', 'request_timeout', 'max_retries')
    @classmethod
    def validate_positive_int(cls, v):
        if v <= 0:
            raise ValueError('Value must be positive')
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",  # Default env file
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        extra="allow",  # Allow extra fields from environment
        # env_prefix="MINDORA_PERFORMANCE_"  # Environment variable prefix
    )