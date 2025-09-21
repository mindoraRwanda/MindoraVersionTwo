from pydantic import BaseModel, EmailStr, constr
from typing import Annotated, List
from datetime import datetime
from backend.app.services.chatbot_insights_pipeline import analyze_user_input




# Type-safe annotated fields
UsernameType = Annotated[str, constr(min_length=3, max_length=20)]
PasswordType = Annotated[str, constr(min_length=6)]

class UserCreate(BaseModel):
    username: UsernameType
    email: EmailStr
    password: PasswordType

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int = None
    username: str = None

class MessageCreate(BaseModel):
    conversation_id: int
    content: Annotated[str, constr(min_length=1)]  # Ensures message is not empty

class MessageOut(BaseModel):
    id: int
    sender: str
    content: str
    timestamp: datetime

    class Config:
        from_attributes = True  # for Pydantic v2 (replaces orm_mode)

class ConversationOut(BaseModel):
    id: int
    started_at: datetime

    class Config:
        from_attributes = True

class EmotionRequest(BaseModel):
    text: str    
        
class AnalysisRequest(BaseModel):
    user_input: str