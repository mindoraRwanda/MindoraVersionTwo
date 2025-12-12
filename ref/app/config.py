"""Configuration management following 12-factor app principles."""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App
    APP_NAME: str = "Mindora API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./mindora.db")
    DATABASE_ECHO: bool = False
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # LLM Configuration
    API_TYPE: str = os.getenv("API_TYPE", "ollama").lower()
    OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    API_BASE_URL: str = os.getenv("API_BASE_URL", os.getenv("OLLAMA_API_URL", "http://localhost:11434"))
    API_KEY: Optional[str] = os.getenv("API_KEY", None)
    
    # Model Configuration
    MODEL: str = os.getenv("MODEL", "qwen2.5:7b")
    SUMMARIZER_MODEL: str = os.getenv("SUMMARIZER_MODEL", "llama3.2:1b")
    SAFETY_MODEL: str = os.getenv("SAFETY_MODEL", "llama-guard3:1b")
    NUM_CTX: int = int(os.getenv("NUM_CTX", "4096"))
    N_TURNS_TO_KEEP: int = int(os.getenv("N_TURNS_TO_KEEP", "10"))
    
    # Context Configuration
    USE_EXTRA_CONTEXT: bool = os.getenv("USE_EXTRA_CONTEXT", "true").lower() == "true"
    
    # KB Configuration
    KB_DIR: str = os.getenv("KB_DIR", "kb/cards")
    KB_RETRIEVAL_K: int = int(os.getenv("KB_RETRIEVAL_K", "1"))  # Number of KB cards to retrieve per turn
    QDRANT_LOCAL_PATH: Optional[str] = os.getenv("QDRANT_LOCAL_PATH", None)
    QDRANT_COLLECTION: str = "mindora_kb"
    EMB_MODEL_NAME: str = os.getenv("EMB_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Experiment / Hyperparameter Configuration
    EXPERIMENT_TAG: str = os.getenv("EXPERIMENT_TAG", "default")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.0"))
    ENABLE_SAFETY: bool = os.getenv("ENABLE_SAFETY", "true").lower() == "true"
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    
    # CORS
    CORS_ORIGINS: str = "*"  # Comma-separated list, or "*" for all
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

