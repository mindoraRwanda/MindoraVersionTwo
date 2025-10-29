from typing import Optional
from .base import BaseAppSettings

class QdrantSettings(BaseAppSettings):
    """Configuration for Qdrant vector database connections."""

    # Qdrant connection settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: Optional[str] = None

    class Config:
        extra = "allow"  # Allow extra fields from environment