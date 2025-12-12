"""Common dependencies."""
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.core.security import get_current_active_user

# Re-export commonly used dependencies
__all__ = ["get_db", "get_current_user"]

# Re-export for convenience
get_current_user = get_current_active_user

