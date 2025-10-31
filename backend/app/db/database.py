
# Base = declarative_base()
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# Use the compatibility layer for gradual migration
from ..settings.settings import settings

# Get database configuration
database_config = settings.database
DATABASE_URL = database_config.database_url if database_config else "postgresql+psycopg2://postgres:12345@localhost:5432/mindora_db"

# Create engine with database settings
engine = create_engine(
    DATABASE_URL,
    pool_size=database_config.database_pool_size if database_config else 5,
    max_overflow=database_config.database_max_overflow if database_config else 10,
    pool_timeout=database_config.database_pool_timeout if database_config else 30,
    pool_recycle=database_config.database_pool_recycle if database_config else 3600
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
