#!/usr/bin/env python3
"""
Database migration script to add the crisis_logs table.

This script creates the crisis_logs table with UUID IDs and all required fields
for crisis detection logging.
"""

import sys
from pathlib import Path
from sqlalchemy import create_engine, text

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from ..settings.settings import settings


def run_migration():
    """Run the migration to add the crisis_logs table."""

    # Get database URL from settings
    database_url = settings.database.database_url if settings.database else "postgresql://user:pass@localhost/db"
    print(f"Connecting to database: {database_url}")

    # Create engine
    engine = create_engine(database_url)

    try:
        print("Starting crisis_logs table migration...")

        with engine.connect() as conn:
            # Create enum types if they don't exist
            print("Creating enum types...")

            # CrisisType enum
            conn.execute(text("""
                DO $$ BEGIN
                    CREATE TYPE crisis_type AS ENUM ('self_harm', 'suicide_ideation', 'self_injury', 'substance_abuse', 'violence', 'medical_emergency', 'other');
                EXCEPTION
                    WHEN duplicate_object THEN null;
                END $$;
            """))

            # CrisisSeverity enum
            conn.execute(text("""
                DO $$ BEGIN
                    CREATE TYPE crisis_severity AS ENUM ('low', 'moderate', 'high', 'imminent');
                EXCEPTION
                    WHEN duplicate_object THEN null;
                END $$;
            """))

            # CrisisStatus enum
            conn.execute(text("""
                DO $$ BEGIN
                    CREATE TYPE crisis_status AS ENUM ('new', 'notified', 'acknowledged', 'resolved', 'false_positive');
                EXCEPTION
                    WHEN duplicate_object THEN null;
                END $$;
            """))

            # Create the crisis_logs table if it doesn't exist
            print("Creating crisis_logs table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS crisis_logs (
                    id INTEGER PRIMARY KEY DEFAULT nextval('crisis_logs_id_seq'),
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
                    message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
                    detected_type crisis_type NOT NULL,
                    severity crisis_severity NOT NULL,
                    confidence FLOAT NOT NULL,
                    excerpt TEXT NOT NULL,
                    rationale TEXT,
                    status crisis_status NOT NULL DEFAULT 'new',
                    classifier_model VARCHAR(120) NOT NULL,
                    classifier_version VARCHAR(40),
                    notified_therapist_id INTEGER REFERENCES therapists(id) ON DELETE SET NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """))

            # Create indexes for performance
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_crisis_logs_user_id ON crisis_logs(user_id);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_crisis_logs_conversation_id ON crisis_logs(conversation_id);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_crisis_logs_message_id ON crisis_logs(message_id);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_crisis_logs_detected_type ON crisis_logs(detected_type);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_crisis_logs_severity ON crisis_logs(severity);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_crisis_logs_status ON crisis_logs(status);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_crisis_logs_created_at ON crisis_logs(created_at);"))

            conn.commit()

        print("✅ Crisis_logs table migration completed successfully!")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    print("⚠️  WARNING: This will add the crisis_logs table to your database.")
    print("⚠️  Please make sure you have a backup of your database before proceeding.")

    response = input("Do you want to continue? (yes/no): ")
    if response.lower() == 'yes':
        run_migration()
    else:
        print("Migration cancelled.")