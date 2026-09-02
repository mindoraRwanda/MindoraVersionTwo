#!/usr/bin/env python3
"""
Database migration script to add Google OAuth support to the users table.

Adds a nullable, unique google_id column (the Google 'sub' claim) and makes
password nullable, since Google-only accounts never set a local password.
"""

from sqlalchemy import text
import sys
import os

# Add the backend directory to the Python path
backend_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_dir)

from app.db.database import engine


def add_google_auth_columns():
    """Add google_id column and relax password NOT NULL on the users table."""
    try:
        with engine.connect() as conn:
            # google_id column
            result = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'google_id'
            """))

            if result.fetchone():
                print("[OK] google_id column already exists")
            else:
                print("Adding google_id column to users table...")
                conn.execute(text("ALTER TABLE users ADD COLUMN google_id VARCHAR(255) UNIQUE"))
                conn.commit()
                print("[OK] Successfully added google_id column")

            # password nullability
            result = conn.execute(text("""
                SELECT is_nullable
                FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'password'
            """))
            row = result.fetchone()

            if row and row[0] == 'YES':
                print("[OK] password column is already nullable")
            else:
                print("Making password column nullable on users table...")
                conn.execute(text("ALTER TABLE users ALTER COLUMN password DROP NOT NULL"))
                conn.commit()
                print("[OK] Successfully made password column nullable")

        return True

    except Exception as e:
        print(f"[ERROR] Error migrating users table for Google auth: {e}")
        return False


if __name__ == "__main__":
    success = add_google_auth_columns()
    if success:
        print("Migration completed successfully!")
        sys.exit(0)
    else:
        print("Migration failed!")
        sys.exit(1)
