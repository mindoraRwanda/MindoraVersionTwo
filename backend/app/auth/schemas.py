from pydantic import BaseModel, EmailStr, constr
from typing import Annotated, List, Optional, Union
from datetime import datetime
# Legacy chatbot_insights_pipeline import removed




# Type-safe annotated fields
UsernameType = Annotated[str, constr(min_length=3, max_length=20)]
PasswordType = Annotated[str, constr(min_length=6)]

class UserCreate(BaseModel):
    username: UsernameType
    email: EmailStr
    password: PasswordType
    gender: Optional[str] = None  # Optional field for gender-based personalization

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: Optional[str] = None  # Changed from UUID to str for compatibility with integer IDs
    username: Optional[str] = None
    gender: Optional[str] = None

class MessageCreate(BaseModel):
    conversation_id: Union[str, int]  # Accept both string and int for compatibility
    content: Annotated[str, constr(min_length=1)]  # Ensures message is not empty

class MessageOut(BaseModel):
    id: str  # Changed from UUID to str for compatibility with integer IDs
    sender: str
    content: str
    timestamp: datetime

    class Config:
        from_attributes = True  # for Pydantic v2 (replaces orm_mode)

class ConversationOut(BaseModel):
    id: str  # Changed from UUID to str for compatibility with integer IDs
    started_at: datetime

    class Config:
        from_attributes = True

class EmotionRequest(BaseModel):
    text: str    
        
class AnalysisRequest(BaseModel):
    user_input: str