from typing import Optional, List, Union, Any
from functools import lru_cache
import os
import json
from pydantic_core import core_schema
from pydantic_settings import SettingsConfigDict
from pydantic import field_validator
from .base import BaseAppSettings, get_environment
from .model import ModelSettings
from .performance import PerformanceSettings
from .safety import SafetySettings
from .cultural import CulturalSettings
from .database import DatabaseSettings
from .emotional import EmotionalResponseSettings
from .qdrant import QdrantSettings


class Settings(BaseAppSettings):
    """Main settings class that aggregates all configuration categories."""
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # API Settings
    api_title: str = "Therapy Chatbot API"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # CORS Settings
    cors_origins: Union[str, List[str]] = "http://localhost:3000"
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from environment variable (JSON or comma-separated)."""
        if isinstance(v, str):
            if not v.strip():  # Empty string
                return ["http://localhost:3000"]
            try:
                # Try to parse as JSON first
                return json.loads(v)
            except json.JSONDecodeError:
                # If not valid JSON, treat as comma-separated string
                return [origin.strip() for origin in v.split(',') if origin.strip()]
        elif isinstance(v, list):
            return v
        else:
            return ["http://localhost:3000"]
    
    # Sub-settings (initialized in __init__)
    model: Optional[ModelSettings] = None
    performance: Optional[PerformanceSettings] = None
    safety: Optional[SafetySettings] = None
    cultural: Optional[CulturalSettings] = None
    database: Optional[DatabaseSettings] = None
    emotional: Optional[EmotionalResponseSettings] = None
    qdrant: Optional[QdrantSettings] = None
    
    model_config = SettingsConfigDict(
        extra="allow"  # Allow extra fields from environment
    )
    
    def __init__(self, **data):
        """Initialize settings with environment-specific configuration."""
        environment = data.get("environment", get_environment())
        
        # Set environment and debug
        data["environment"] = environment
        
        # Handle debug environment variable properly
        debug_env = os.getenv("DEBUG")
        if debug_env is not None:
            # Convert string to boolean properly
            data["debug"] = debug_env.lower() in ("true", "1", "yes", "on")
        else:
            # Default based on environment
            data["debug"] = environment in ["development", "testing"]
        
        super().__init__(**data)
        
        # Initialize sub-settings with environment
        self.model = ModelSettings.create_for_environment(environment)
        self.performance = PerformanceSettings.create_for_environment(environment)
        self.safety = SafetySettings.create_for_environment(environment)
        self.cultural = CulturalSettings.create_for_environment(environment)
        self.database = DatabaseSettings.create_for_environment(environment)
        self.emotional = EmotionalResponseSettings.create_for_environment(environment)
        self.qdrant = QdrantSettings.create_for_environment(environment)

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Global settings instance
settings = get_settings()