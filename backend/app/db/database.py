# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker, declarative_base

# DATABASE_URL = "postgresql://postgres:12345@localhost:5100/postgres"
# # Print to confirm connection string
# print(f"Connecting to database: {DATABASE_URL}")
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base = declarative_base()
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

#DATABASE_URL = "postgresql+psycopg2://postgres:12345@localhost:5100/mindora_db"
DATABASE_URL="postgresql+psycopg2://postgres:12345@localhost:5432/mindora_db"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
