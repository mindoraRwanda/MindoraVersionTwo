from typing import Any, Dict, Optional, Type, TypeVar, Union
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings as PydanticBaseSettings, SettingsConfigDict
import os
from functools import lru_cache

T = TypeVar('T', bound='BaseAppSettings')

class BaseAppSettings(PydanticBaseSettings):
    """Base settings class with common functionality for all configuration categories."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True
    )
        
    @classmethod
    def create_for_environment(cls: Type[T], environment: str) -> T:
        """Create settings instance for a specific environment.

        Simpler and more robust: prefer passing `_env_file` to the constructor
        which is supported by pydantic-settings instead of creating dynamic subclasses.
        """
        env_file = f".env.{environment}"
        if Path(env_file).exists():
            # Use the per-instance env file override supported by pydantic settings
            return cls(_env_file=env_file)
        return cls()
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export settings to dictionary format."""
        return self.dict()
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update settings from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

@lru_cache()
def get_environment() -> str:
    """Get current environment from ENVIRONMENT variable or default to development."""
    return os.getenv("ENVIRONMENT", "development").lower()