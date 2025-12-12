"""Authentication schemas."""
from pydantic import BaseModel, EmailStr, Field, field_serializer
from typing import Optional, Dict, Any, List
from datetime import datetime


class UserSignup(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    sex: Optional[str] = None
    age_years: Optional[int] = None
    religion: Optional[str] = None
    languages: List[str] = ["en"]
    location: Optional[Dict[str, Any]] = None
    time_zone: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    sex: Optional[str] = None
    age_years: Optional[int] = None
    religion: Optional[str] = None
    languages: Optional[List[str]] = None
    location: Optional[Dict[str, Any]] = None
    time_zone: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    first_name: Optional[str]
    last_name: Optional[str]
    sex: Optional[str]
    age_years: Optional[int]
    religion: Optional[str]
    languages: List[str]
    location: Dict[str, Any]
    time_zone: Optional[str]
    is_active: bool
    created_at: datetime
    
    @field_serializer('created_at')
    def serialize_datetime(self, dt: datetime, _info) -> str:
        """Serialize datetime to ISO format string."""
        if dt is None:
            return None
        # isoformat() handles timezone-aware datetimes correctly
        return dt.isoformat()
    
    class Config:
        from_attributes = True

