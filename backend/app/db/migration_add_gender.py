#!/usr/bin/env python3
"""
Database migration script to add gender column to users table.

This script adds a gender field to the existing users table for gender-aware personalization.
"""

from sqlalchemy import create_engine, Column, String, Integer, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.app.db.database import DATABASE_URL

def add_gender_column():
    """Add gender column to users table."""

    # Create engine
    engine = create_engine(DATABASE_URL)

    # Create a base for the migration
    Base = declarative_base()

    # Define a temporary User model for migration
    class User(Base):
        __tablename__ = 'users'
        id = Column(Integer, primary_key=True)
        gender = Column(String(20), nullable=True)

    try:
        # Add the gender column
        print("Adding gender column to users table...")
        with engine.connect() as conn:
            # Check if column already exists
            result = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'gender'
            """))

            if result.fetchone():
                print("✅ Gender column already exists")
                return True

            # Add the gender column
            conn.execute(text("ALTER TABLE users ADD COLUMN gender VARCHAR(20)"))
            conn.commit()

        print("✅ Successfully added gender column to users table")
        return True

    except Exception as e:
        print(f"❌ Error adding gender column: {e}")
        return False

if __name__ == "__main__":
    success = add_gender_column()
    if success:
        print("Migration completed successfully!")
        sys.exit(0)
    else:
        print("Migration failed!")
        sys.exit(1)