#!/usr/bin/env python3
"""
Database migration script to add gender column to users table.

This script adds a gender field to the existing users table for gender-aware personalization.
Works with both SQLite and PostgreSQL.
"""

from sqlalchemy import text
import sys
from pathlib import Path

# Add the project root to the Python path
# From backend/app/db/migration_add_gender.py -> go up 3 levels to project root
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.db.database import DATABASE_URL, engine

def check_column_exists_sqlite(conn):
    """Check if gender column exists in SQLite."""
    result = conn.execute(text("PRAGMA table_info(users)"))
    columns = result.fetchall()
    for col in columns:
        if len(col) >= 2 and col[1] == 'gender':  # col[1] is column name
            return True
    return False

def check_column_exists_postgres(conn):
    """Check if gender column exists in PostgreSQL."""
    result = conn.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'gender'
    """))
    return result.fetchone() is not None

def add_gender_column():
    """Add gender column to users table."""
    
    # Detect database type from URL
    is_sqlite = 'sqlite' in DATABASE_URL.lower()
    
    try:
        print(f"🔍 Detected database type: {'SQLite' if is_sqlite else 'PostgreSQL'}")
        print("Adding gender column to users table...")
        
        with engine.begin() as conn:  # begin() automatically commits on success
            # Check if column already exists
            if is_sqlite:
                column_exists = check_column_exists_sqlite(conn)
            else:
                column_exists = check_column_exists_postgres(conn)
            
            if column_exists:
                print("✅ Gender column already exists")
                return True
            
            # Add the gender column (SQLite and PostgreSQL both support this syntax)
            if is_sqlite:
                # SQLite syntax - no need to specify VARCHAR length for SQLite
                conn.execute(text("ALTER TABLE users ADD COLUMN gender TEXT"))
            else:
                # PostgreSQL syntax
                conn.execute(text("ALTER TABLE users ADD COLUMN gender VARCHAR(20)"))

        print("✅ Successfully added gender column to users table")
        return True

    except Exception as e:
        print(f"❌ Error adding gender column: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = add_gender_column()
    if success:
        print("✅ Migration completed successfully!")
        sys.exit(0)
    else:
        print("❌ Migration failed!")
        sys.exit(1)