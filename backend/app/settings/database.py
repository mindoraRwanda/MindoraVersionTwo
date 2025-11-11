from typing import Optional
from pydantic import field_validator, ConfigDict
from .base import BaseAppSettings

class DatabaseSettings(BaseAppSettings):
    """Configuration for database connections."""
    
    # Primary database
    database_url: str = "sqlite:///./mindora.db"
    
    # Database connection settings
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    
    # Redis settings (for caching)
    redis_url: Optional[str] = None
    redis_ttl: int = 3600
    
    @field_validator('database_pool_size', 'database_max_overflow', 'database_pool_timeout',
                    'database_pool_recycle', 'redis_ttl')
    def validate_positive_int(cls, v):
        if v <= 0:
            raise ValueError('Value must be positive')
        return v
    
    # Pydantic V2 model configuration
    model_config = ConfigDict(extra="allow")  # Allow extra fields from environment