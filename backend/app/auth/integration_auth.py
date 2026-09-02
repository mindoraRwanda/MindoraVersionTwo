"""
Integration auth for the Mindora chatbot.

The chatbot no longer signs users up or logs them in. Instead it exposes a
single token-exchange endpoint (see routers/integration_router.py) that the main
web app's BACKEND calls, server-to-server, for a user it has already
authenticated. The chatbot answers with a short-lived access token, and that
token is what the user's browser then sends to the chatbot.

    main app backend  --(X-Integration-Key + user info)-->  POST /integration/session
    main app backend  <--(chatbot access token, 30 min)---
    main app frontend --(Authorization: Bearer <that token>)--> /auth/messages, ...

Two secrets, doing different jobs:

  INTEGRATION_API_KEY        Long-lived. Handed to the main app's team. Proves a
                             caller is that backend and is allowed to mint tokens
                             for any user. Server-to-server only — if this ever
                             reaches a browser, anyone can impersonate anyone.

  INTEGRATION_TOKEN_SECRET   Signs the access tokens. Never leaves this service;
                             the main app never needs it and must never have it.

That split is why one key alone is not enough: the API key says "this is the main
app", the access token says "this is Alice". Without the second, any holder of the
first could read anyone's conversations — which here includes crisis logs.
"""

import hmac
import logging
import os
import re
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import jwt as pyjwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..db.models import User

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)

INTEGRATION_ALGORITHM = "HS256"

# ---------------------------------------------------------------------------
# Environment loading
#
# main.py loads the .env file at startup, but this module must not depend on
# that having happened — import order is not guaranteed, and a value read too
# early is silently empty, which surfaces as an unexplained 503. So if the
# integration variables are absent when first needed, find and load the .env
# ourselves, once.
#
# override=False on purpose: a value already in the real process environment
# (Railway, Docker, an export in the shell) always wins over the file. That
# keeps production safe from a stale .env sitting in the image.
# ---------------------------------------------------------------------------

_env_loaded = False
_env_source = None


def _ensure_env_loaded() -> None:
    global _env_loaded, _env_source

    if _env_loaded:
        return
    _env_loaded = True

    if os.getenv("INTEGRATION_API_KEY") or os.getenv("INTEGRATION_TOKEN_SECRET"):
        _env_source = "process environment"
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv is not installed — relying on the process environment")
        return

    # backend/app/auth/integration_auth.py -> repository root
    root = Path(__file__).resolve().parents[3]
    environment = os.getenv("ENVIRONMENT", "development").lower()

    for candidate in (root / f".env.{environment}", root / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            _env_source = str(candidate)
            logger.info("Integration settings loaded from %s", candidate)
            return

    logger.warning("No .env file found under %s — integration variables must come "
                   "from the process environment", root)


# These are read on every use, not once at import time. Import order is not
# guaranteed to run after the .env file is loaded, and a value captured too
# early is silently empty — which surfaces as a confusing 503 rather than an
# error at startup. Reading late also means editing .env and reloading picks
# the change up without a full restart.

def _api_key() -> str:
    """Shared key proving a caller is the main app's backend."""
    _ensure_env_loaded()
    return os.getenv("INTEGRATION_API_KEY", "")


def _token_secret() -> str:
    """Signing key for access tokens. Never leaves this service."""
    _ensure_env_loaded()
    return os.getenv("INTEGRATION_TOKEN_SECRET", "")


def _expire_minutes() -> int:
    try:
        return int(os.getenv("INTEGRATION_TOKEN_EXPIRE_MINUTES", "30"))
    except ValueError:
        logger.warning("INTEGRATION_TOKEN_EXPIRE_MINUTES is not a number — using 30")
        return 30


def _auto_link_by_email() -> bool:
    """Claim a pre-integration chatbot account when the email matches, so
    existing users keep their conversations. Only safe if the MAIN APP
    verifies email addresses."""
    _ensure_env_loaded()
    _ensure_env_loaded()
    return os.getenv("INTEGRATION_AUTO_LINK_BY_EMAIL", "true").lower() == "true"


def _allow_legacy_tokens() -> bool:
    """Keep accepting tokens from the chatbot's own /auth/login during cutover."""
    _ensure_env_loaded()
    return os.getenv("INTEGRATION_ALLOW_LEGACY_TOKENS", "true").lower() == "true"


def integration_status() -> dict:
    """Startup diagnostic — reports what is configured without leaking values."""
    key, secret = _api_key(), _token_secret()
    src = _env_source or "not loaded"
    return {
        "api_key_set": bool(key),
        "api_key_length": len(key),
        "token_secret_set": bool(secret),
        "token_expire_minutes": _expire_minutes(),
        "auto_link_by_email": _auto_link_by_email(),
        "allow_legacy_tokens": _allow_legacy_tokens(),
        "source": src,
    }

TOKEN_TYPE = "integration_access"

credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def _reject(reason: str) -> HTTPException:
    """Log why, tell the caller nothing useful."""
    logger.warning("Integration auth rejected: %s", reason)
    return credentials_exception


# --------------------------------------------------------------------------
# 1. Who may mint tokens
# --------------------------------------------------------------------------

def require_integration_key(x_integration_key: Optional[str] = Header(default=None)) -> None:
    """Guard the token-exchange endpoint. Server-to-server callers only."""
    api_key = _api_key()
    if not api_key:
        logger.error("INTEGRATION_API_KEY is not set — the integration endpoint is disabled")
        raise HTTPException(status_code=503, detail="Integration is not configured")
    # Constant-time so the key can't be recovered by timing responses.
    if not x_integration_key or not hmac.compare_digest(x_integration_key, api_key):
        logger.warning("Integration endpoint called with a missing or wrong API key")
        raise HTTPException(status_code=403, detail="Invalid integration key")


# --------------------------------------------------------------------------
# 2. Mapping a main-app user onto a local users row
# --------------------------------------------------------------------------

def _unique_username(db: Session, base: str) -> str:
    """Derive a username satisfying the existing 3-20 character constraint."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "", base or "")[:20]
    if len(cleaned) < 3:
        cleaned = (cleaned + "user")[:20]
    candidate, suffix = cleaned, 0
    while db.query(User).filter(User.username == candidate).first():
        suffix += 1
        s = str(suffix)
        candidate = f"{cleaned[:20 - len(s)]}{s}"
    return candidate


def provision_user(
    db: Session,
    external_id: str,
    email: str,
    username: Optional[str] = None,
    gender: Optional[str] = None,
) -> User:
    """
    Find or create the local record for a main-app user. Called once per token
    exchange, so the users table stays in sync with the main app without any
    polling, batch import, or scheduled job — a user appears here the first time
    they open the chat, and never before.

    Resolution order:
      1. external_id  — the normal path.
      2. email        — one-time claim of a pre-integration chatbot account, so
                        its conversations, emotion logs and crisis history carry
                        over. Stamps external_id so step 1 handles it next time.
      3. create       — someone who never used the standalone chatbot.
    """
    external_id = str(external_id).strip()
    email = (email or "").strip().lower()
    if not external_id:
        raise HTTPException(status_code=422, detail="external_id is required")

    user = db.query(User).filter(User.external_id == external_id).first()
    if user:
        # Keep the mirrored fields fresh — the main app owns them.
        changed = False
        if email and user.email != email:
            clash = db.query(User).filter(User.email == email, User.id != user.id).first()
            if clash:
                logger.warning(
                    "Cannot update user %s to email %s — already used by %s", user.id, email, clash.id
                )
            else:
                user.email, changed = email, True
        if gender and user.gender != gender:
            user.gender, changed = gender, True
        if changed:
            db.commit()
            db.refresh(user)
        return user

    if email and _auto_link_by_email():
        legacy = db.query(User).filter(User.email == email).first()
        if legacy:
            if legacy.external_id and legacy.external_id != external_id:
                # Two main-app accounts claiming one chatbot account. Refuse rather
                # than silently hand over someone else's chat history.
                logger.error(
                    "Refusing to relink %s: already bound to external_id %s",
                    email, legacy.external_id,
                )
                raise HTTPException(
                    status_code=409,
                    detail="This email is already linked to a different account",
                )
            legacy.external_id = external_id
            legacy.auth_provider = "main_app"
            if gender and not legacy.gender:
                legacy.gender = gender
            db.commit()
            db.refresh(legacy)
            logger.info("Linked existing chatbot account %s to main-app user %s", legacy.id, external_id)
            return legacy

    if not email:
        raise HTTPException(status_code=422, detail="email is required to create a chatbot profile")

    new_user = User(
        id=uuid.uuid4(),
        uuid=uuid.uuid4(),
        external_id=external_id,
        auth_provider="main_app",
        username=_unique_username(db, username or email.split("@")[0]),
        email=email,
        password=None,          # integrated accounts never hold a password
        gender=gender,
    )
    db.add(new_user)
    try:
        db.commit()
    except Exception:
        # Two concurrent first requests for the same user can race here.
        db.rollback()
        existing = db.query(User).filter(User.external_id == external_id).first()
        if existing:
            return existing
        raise
    db.refresh(new_user)
    logger.info("Provisioned chatbot profile %s for main-app user %s", new_user.id, external_id)
    return new_user


# --------------------------------------------------------------------------
# 3. Issuing and verifying the access token
# --------------------------------------------------------------------------

def issue_access_token(user: User) -> dict:
    """Mint the token the main app hands to its frontend."""
    secret = _token_secret()
    if not secret:
        logger.error("INTEGRATION_TOKEN_SECRET is not set — cannot issue access tokens")
        raise HTTPException(status_code=503, detail="Integration is not configured")

    now = datetime.utcnow()
    expire_minutes = _expire_minutes()
    expires_at = now + timedelta(minutes=expire_minutes)
    payload = {
        "sub": str(user.uuid),
        "ext": user.external_id,
        "token_type": TOKEN_TYPE,
        "iat": now,
        "exp": expires_at,
        "jti": str(uuid.uuid4()),
    }
    token = pyjwt.encode(payload, secret, algorithm=INTEGRATION_ALGORITHM)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": expire_minutes * 60,
        "user_id": user.uuid,
        "username": user.username,
        "gender": user.gender,
    }


def _decode_access_token(token: str) -> dict:
    secret = _token_secret()
    if not secret:
        raise HTTPException(status_code=503, detail="Integration is not configured")
    try:
        payload = pyjwt.decode(
            token,
            secret,
            algorithms=[INTEGRATION_ALGORITHM],
            options={"require": ["exp", "iat", "sub"]},
            leeway=30,  # tolerate small clock drift between hosts
        )
    except pyjwt.ExpiredSignatureError:
        raise _reject("access token expired")
    except pyjwt.PyJWTError as e:
        raise _reject(f"invalid access token ({type(e).__name__})")

    if payload.get("token_type") != TOKEN_TYPE:
        raise _reject("wrong token type")
    return payload


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Drop-in replacement for the old auth.utils.get_current_user.

    Accepts an integration access token. While INTEGRATION_ALLOW_LEGACY_TOKENS is
    on, a token from the chatbot's own /auth/login still works too, so existing
    sessions survive the deploy. Turn it off once the standalone login is retired.
    """
    if credentials is None or not credentials.credentials:
        raise credentials_exception

    token = credentials.credentials

    # Peek without verifying, only to route to the right verifier.
    try:
        unverified = pyjwt.decode(token, options={"verify_signature": False})
    except pyjwt.PyJWTError:
        raise _reject("not a JWT")

    if unverified.get("token_type") == TOKEN_TYPE:
        payload = _decode_access_token(token)
        user = db.query(User).filter(User.uuid == uuid.UUID(payload["sub"])).first()
        if user is None:
            raise _reject("token references a deleted user")
        return user

    if _allow_legacy_tokens():
        from .utils import get_current_user_legacy
        return get_current_user_legacy(token, db)

    raise _reject("legacy chatbot tokens are no longer accepted")
