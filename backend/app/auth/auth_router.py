from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .schemas import UserCreate, UserLogin, TokenResponse
from .utils import hash_password, verify_password, create_access_token
from ..db.database import SessionLocal
from ..db.models import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

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

    print(f"Login successful for user: {data}")

    return data




