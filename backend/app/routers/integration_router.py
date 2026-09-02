"""
The endpoint the main web app calls to get a chatbot access key for one of its
already-authenticated users.

This is the whole integration surface. The main app's team never touches the
chatbot database, never learns the token-signing secret, and only ever calls
one route.

    POST /integration/session
    X-Integration-Key: <the key you gave them>
    Content-Type: application/json

    { "external_id": "41", "email": "alice@example.com",
      "username": "alice", "gender": "female" }

    200 OK
    { "access_token": "eyJ...", "token_type": "bearer", "expires_in": 1800,
      "user_id": "…", "username": "alice", "gender": "female" }

The main app's BACKEND makes this call (the key must never reach a browser),
then hands the returned access_token to its own frontend, which sends it to the
chatbot as a normal `Authorization: Bearer` header.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from ..auth.integration_auth import (
    _expire_minutes,
    get_current_user,
    issue_access_token,
    provision_user,
    require_integration_key,
)
from ..db.database import get_db
from ..db.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integration", tags=["Main app integration"])


class SessionRequest(BaseModel):
    # The main app's own user id. Must be STABLE for the life of the account —
    # the chatbot keys all chat history off it, so an id that changes orphans
    # someone's conversations.
    external_id: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    username: Optional[str] = Field(default=None, max_length=100)
    gender: Optional[str] = Field(default=None, max_length=20)


class SessionResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    username: str
    gender: Optional[str] = None


@router.post(
    "/session",
    response_model=SessionResponse,
    dependencies=[Depends(require_integration_key)],
    summary="Exchange a main-app user for a chatbot access token",
)
def create_session(payload: SessionRequest, db: Session = Depends(get_db)):
    """
    Issue a short-lived chatbot access token for a user the caller has already
    authenticated. Creates the local chatbot profile on first call — there is no
    separate import or sync step.

    Call this when the user opens the chat, and again whenever the previous token
    is close to expiry. It is cheap and idempotent.
    """
    user = provision_user(
        db=db,
        external_id=payload.external_id,
        email=payload.email,
        username=payload.username,
        gender=payload.gender,
    )
    token_data = issue_access_token(user)
    logger.info(
        "Issued chatbot access token for main-app user %s (chatbot user %s), valid %d min",
        payload.external_id, user.id, _expire_minutes(),
    )
    return {
        "access_token": token_data["access_token"],
        "token_type": "bearer",
        "expires_in": token_data["expires_in"],
        "user_id": str(user.uuid),
        "username": user.username,
        "gender": user.gender,
    }


@router.get("/me", summary="Who does this access token belong to?")
def whoami(user: User = Depends(get_current_user)):
    """
    Lets the main app's frontend confirm a token is still valid without sending a
    message. Useful for a health check during integration testing.
    """
    return {
        "user_id": str(user.uuid),
        "external_id": user.external_id,
        "username": user.username,
        "email": user.email,
        "gender": user.gender,
    }
