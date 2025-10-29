#!/usr/bin/env python3
"""
Migration script to move data from SQLite to PostgreSQL.
This script reads all data from the SQLite database and inserts it into PostgreSQL.
"""

import sqlite3
import psycopg2
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../.env.development')

def get_sqlite_connection():
    """Get SQLite database connection."""
    return sqlite3.connect('dev.db')

def get_postgres_connection():
    """Get PostgreSQL database connection."""
    return psycopg2.connect(
        host="localhost",
        database="mindora",
        user="postgres",
        password=""  # Default PostgreSQL setup
    )

def migrate_table(cursor_sqlite, cursor_postgres, table_name, columns):
    """Migrate data from one table."""
    print(f"Migrating table: {table_name}")

    # Get data from SQLite
    cursor_sqlite.execute(f"SELECT {', '.join(columns)} FROM {table_name}")
    rows = cursor_sqlite.fetchall()

    if not rows:
        print(f"  No data to migrate in {table_name}")
        return

    # Insert data into PostgreSQL
    placeholders = ', '.join(['%s'] * len(columns))
    query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

    cursor_postgres.executemany(query, rows)
    print(f"  Migrated {len(rows)} rows from {table_name}")

def main():
    """Main migration function."""
    print("Starting SQLite to PostgreSQL migration...")

    # Connect to databases
    conn_sqlite = get_sqlite_connection()
    conn_postgres = get_postgres_connection()

    cursor_sqlite = conn_sqlite.cursor()
    cursor_postgres = conn_postgres.cursor()

    try:
        # Create tables in PostgreSQL first
        from app.db.models import Base
        Base.metadata.create_all(bind=conn_postgres)

        print("PostgreSQL tables created.")

        # Migrate data table by table
        # Users table
        migrate_table(cursor_sqlite, cursor_postgres, 'users',
                     ['id', 'uuid', 'username', 'email', 'password', 'gender', 'created_at'])

        # Conversations table
        migrate_table(cursor_sqlite, cursor_postgres, 'conversations',
                     ['id', 'uuid', 'user_id', 'started_at', 'last_activity_at'])

        # Messages table
        migrate_table(cursor_sqlite, cursor_postgres, 'messages',
                     ['id', 'uuid', 'conversation_id', 'sender', 'content', 'timestamp'])

        # Emotion logs table
        migrate_table(cursor_sqlite, cursor_postgres, 'emotion_logs',
                     ['id', 'uuid', 'user_id', 'conversation_id', 'input_text', 'detected_emotion', 'timestamp'])

        # Commit changes
        conn_postgres.commit()
        print("✅ Migration completed successfully!")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn_postgres.rollback()
        raise
    finally:
        conn_sqlite.close()
        conn_postgres.close()

if __name__ == "__main__":
    main()