#!/usr/bin/env python3
"""
Script to recreate PostgreSQL tables with proper UUID schema.
This drops existing tables and recreates them with the correct schema.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.app.db.models import Base
from backend.app.db.database import engine
from sqlalchemy import text

def recreate_tables():
    """Drop existing tables and recreate with proper UUID schema."""
    print("Recreating PostgreSQL tables with UUID schema...")

    try:
        # Drop tables in correct order (reverse dependency order)
        with engine.connect() as conn:
            # Disable foreign key checks temporarily
            conn.execute(text("SET CONSTRAINTS ALL DEFERRED"))

            # Drop tables
            conn.execute(text("DROP TABLE IF EXISTS emotion_logs CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS messages CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS conversations CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS users CASCADE"))

            conn.commit()

        print("✅ Dropped existing tables")

        # Recreate tables with proper schema
        Base.metadata.create_all(bind=engine)
        print("✅ Recreated tables with UUID schema")

        # Verify tables have UUID columns
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_name IN ('users', 'conversations', 'messages', 'emotion_logs')
                AND column_name = 'uuid'
                ORDER BY table_name
            """))

            uuid_columns = result.fetchall()
            print(f"✅ Verified UUID columns: {len(uuid_columns)} found")
            for table, column in uuid_columns:
                print(f"  - {table}.{column}")

    except Exception as e:
        print(f"❌ Error recreating tables: {e}")
        raise

if __name__ == "__main__":
    recreate_tables()