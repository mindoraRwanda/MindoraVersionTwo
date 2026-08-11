import os
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .schemas import (
    UserCreate, UserLogin, TokenResponse,
    ForgotPasswordRequest, ResetPasswordRequest, MessageResponse,
)
from .utils import (
    hash_password, verify_password, create_access_token,
    create_reset_token, decode_reset_token, RESET_TOKEN_EXPIRE_MINUTES,
)
from ..db.database import SessionLocal
from ..db.models import User
from ..services.emailer import send_email, render_password_reset_email
from ..settings.base import get_environment

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

if get_environment() == "production" and "localhost" in FRONTEND_URL:
    # FRONTEND_URL is set per-environment on the host (e.g. Render dashboard),
    # not in the committed .env — if this fires, password-reset links are
    # being emailed to real users pointing at localhost. Loud on purpose.
    logger.error(
        f"⚠️ FRONTEND_URL is '{FRONTEND_URL}' in a production environment — "
        f"password-reset emails will contain broken localhost links. "
        f"Set FRONTEND_URL to the real deployed frontend URL in this environment's config."
    )

# Dependency: get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Authentication Endpoints ---

@router.post("/signup", response_model=TokenResponse)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account."""
    # Check for existing email
    existing_email = db.query(User).filter(User.email == user.email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already in use")

    # Check for existing username
    existing_username = db.query(User).filter(User.username == user.username).first()
    if existing_username:
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed_pw = hash_password(user.password)
    new_user = User(username=user.username, email=user.email, password=hashed_pw, gender=user.gender)

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token(data={"sub": str(new_user.uuid)})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": new_user.uuid,
        "username": new_user.username,
        "gender": new_user.gender
    }


@router.post("/login", response_model=TokenResponse)
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return access token."""
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": str(user.uuid)})
    data =  {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.uuid,
        "username": user.username,
        "gender": user.gender
    }

    # Never log the token itself — logging user_id only is enough to trace a
    # login without putting a usable credential in the logs.
    logger.info(f"Login successful for user_id={user.uuid}")

    return data


@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Request a password-reset link. Always returns the same generic message
    regardless of whether the email is registered — returning a different
    message for unknown emails would let this endpoint be used to check
    which addresses have accounts.
    """
    generic_response = {"message": "If that email is registered, a reset link has been sent."}

    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        logger.info(f"Password reset requested for unregistered email: {payload.email}")
        return generic_response

    reset_token = create_reset_token(str(user.uuid))
    reset_link = f"{FRONTEND_URL}/reset-password?token={reset_token}"

    # Never log the reset link/token, in any environment — it's a live
    # credential that can take over the account until it expires. Delivery
    # is verified via the sent/failed log below instead.
    logger.info(f"[password reset] link generated for {user.email}")

    subject, text_body, html_body = render_password_reset_email(reset_link, RESET_TOKEN_EXPIRE_MINUTES)
    sent = send_email(to_email=user.email, subject=subject, text=text_body, html=html_body)
    if sent:
        logger.info(f"[password reset] email sent to {user.email}")
    else:
        logger.error(
            f"[password reset] email FAILED to send to {user.email} — check SMTP_HOST/PORT/USER/PASS "
            f"and the [email failed] log line above for the underlying error."
        )

    return generic_response


@router.post("/reset-password", response_model=MessageResponse)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Complete a password reset using the token issued by /forgot-password."""
    try:
        user_uuid = decode_reset_token(payload.token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    user = db.query(User).filter(User.uuid == user_uuid).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")

    user.password = hash_password(payload.new_password)
    db.commit()

    return {"message": "Password reset successful. You can now log in with your new password."}

