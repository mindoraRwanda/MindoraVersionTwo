import bcrypt
import jwt as pyjwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from uuid import UUID
from ..db.models import User
from ..db.database import get_db
import os
import hashlib

# Auth scheme for FastAPI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "dev_default_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 360

def hash_password(password: str) -> str:
    # Pre-hash with SHA-256 to handle any length password while preserving uniqueness
    # This ensures we stay within bcrypt's 72-byte limit without losing password data
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    # Use bcrypt directly to avoid passlib compatibility issues
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_hash.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Pre-hash with SHA-256 to match the hashing process
    password_hash = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
    # Use bcrypt directly to avoid passlib compatibility issues
    return bcrypt.checkpw(password_hash.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "token_type": "access"})
    return pyjwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except pyjwt.PyJWTError:
        raise credentials_exception

    user = db.query(User).filter(User.uuid == UUID(user_id)).first()
    if user is None:
        raise credentials_exception
    return user
