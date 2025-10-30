#!/usr/bin/env python3
"""
Database migration script to convert integer IDs to UUIDs.

This script will:
1. Create new UUID columns for all tables
2. Populate the UUID columns with new UUID values
3. Update foreign key references
4. Drop old integer ID columns
5. Update primary key constraints

IMPORTANT: This migration should be run with caution and with a database backup!
"""

import uuid
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from ..settings.settings import settings
from ..db.models import Base, User, Conversation, Message, EmotionLog


def run_migration():
    """Run the UUID migration."""
    
    # Get database URL from settings
    database_url = settings.database.database_url if settings.database else "sqlite:///./dev.db"
    print(f"Connecting to database: {database_url}")
    
    # Create engine
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        print("Starting UUID migration...")
        
        # Check if migration has already been run
        try:
            result = session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'id_new'"))
            if result.fetchone():
                print("Migration appears to have already been started. Please check the database state.")
                return
        except Exception:
            # For SQLite, the information_schema query won't work
            pass
        
        # For SQLite, we need to recreate tables since it doesn't support ALTER COLUMN well
        if "sqlite" in database_url.lower():
            migrate_sqlite(session, engine)
        else:
            # For PostgreSQL and other databases
            migrate_postgresql(session, engine)
        
        print("✅ UUID migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def migrate_sqlite(session, engine):
    """Migration for SQLite (in-place migration)."""
    print("Running SQLite migration...")

    # Global mappings for ID conversion
    old_id_to_uuid = {}
    conv_old_id_to_uuid = {}

    # 1. Add new UUID columns to existing tables
    print("Adding UUID columns to existing tables...")

    # Add UUID columns
    session.execute(text("ALTER TABLE users ADD COLUMN id_new TEXT"))
    session.execute(text("ALTER TABLE conversations ADD COLUMN id_new TEXT"))
    session.execute(text("ALTER TABLE conversations ADD COLUMN user_id_new TEXT"))
    session.execute(text("ALTER TABLE messages ADD COLUMN id_new TEXT"))
    session.execute(text("ALTER TABLE messages ADD COLUMN conversation_id_new TEXT"))
    session.execute(text("ALTER TABLE emotion_logs ADD COLUMN id_new TEXT"))
    session.execute(text("ALTER TABLE emotion_logs ADD COLUMN user_id_new TEXT"))
    session.execute(text("ALTER TABLE emotion_logs ADD COLUMN conversation_id_new TEXT"))

    session.commit()

    # 2. Populate UUID columns with new values
    print("Populating UUID columns...")

    # Generate UUIDs for users
    users = session.execute(text("SELECT id FROM users")).fetchall()
    for user in users:
        old_id = user[0]
        new_uuid = str(uuid.uuid4())
        old_id_to_uuid[old_id] = new_uuid
        session.execute(text("UPDATE users SET id_new = ? WHERE id = ?"), (new_uuid, old_id))

    # Generate UUIDs for conversations and update user references
    conversations = session.execute(text("SELECT id, user_id FROM conversations")).fetchall()
    for conv in conversations:
        old_id = conv[0]
        old_user_id = conv[1]
        new_uuid = str(uuid.uuid4())
        user_uuid = old_id_to_uuid.get(old_user_id)

        if user_uuid:
            conv_old_id_to_uuid[old_id] = new_uuid
            session.execute(text("UPDATE conversations SET id_new = ?, user_id_new = ? WHERE id = ?"),
                          (new_uuid, user_uuid, old_id))

    # Generate UUIDs for messages and update conversation references
    messages = session.execute(text("SELECT id, conversation_id FROM messages")).fetchall()
    for msg in messages:
        old_id = msg[0]
        old_conv_id = msg[1]
        new_uuid = str(uuid.uuid4())
        conv_uuid = conv_old_id_to_uuid.get(old_conv_id)

        if conv_uuid:
            session.execute(text("UPDATE messages SET id_new = ?, conversation_id_new = ? WHERE id = ?"),
                          (new_uuid, conv_uuid, old_id))

    # Generate UUIDs for emotion logs and update references
    emotion_logs = session.execute(text("SELECT id, user_id, conversation_id FROM emotion_logs")).fetchall()
    for log in emotion_logs:
        old_id = log[0]
        old_user_id = log[1]
        old_conv_id = log[2]
        new_uuid = str(uuid.uuid4())
        user_uuid = old_id_to_uuid.get(old_user_id)
        conv_uuid = conv_old_id_to_uuid.get(old_conv_id)

        if user_uuid and conv_uuid:
            session.execute(text("UPDATE emotion_logs SET id_new = ?, user_id_new = ?, conversation_id_new = ? WHERE id = ?"),
                          (new_uuid, user_uuid, conv_uuid, old_id))

    session.commit()

    # 3. Drop old columns and rename new ones
    print("Dropping old columns and renaming new ones...")

    # Drop old columns
    session.execute(text("ALTER TABLE users DROP COLUMN id"))
    session.execute(text("ALTER TABLE conversations DROP COLUMN id"))
    session.execute(text("ALTER TABLE conversations DROP COLUMN user_id"))
    session.execute(text("ALTER TABLE messages DROP COLUMN id"))
    session.execute(text("ALTER TABLE messages DROP COLUMN conversation_id"))
    session.execute(text("ALTER TABLE emotion_logs DROP COLUMN id"))
    session.execute(text("ALTER TABLE emotion_logs DROP COLUMN user_id"))
    session.execute(text("ALTER TABLE emotion_logs DROP COLUMN conversation_id"))

    # Rename new columns
    session.execute(text("ALTER TABLE users RENAME COLUMN id_new TO id"))
    session.execute(text("ALTER TABLE conversations RENAME COLUMN id_new TO id"))
    session.execute(text("ALTER TABLE conversations RENAME COLUMN user_id_new TO user_id"))
    session.execute(text("ALTER TABLE messages RENAME COLUMN id_new TO id"))
    session.execute(text("ALTER TABLE messages RENAME COLUMN conversation_id_new TO conversation_id"))
    session.execute(text("ALTER TABLE emotion_logs RENAME COLUMN id_new TO id"))
    session.execute(text("ALTER TABLE emotion_logs RENAME COLUMN user_id_new TO user_id"))
    session.execute(text("ALTER TABLE emotion_logs RENAME COLUMN conversation_id_new TO conversation_id"))

    session.commit()
    print("✅ Data migration completed for SQLite")


def migrate_postgresql(session, engine):
    """Migration for PostgreSQL (supports ALTER COLUMN)."""
    print("Running PostgreSQL migration...")
    
    # Global mappings for ID conversion
    old_id_to_uuid = {}
    conv_old_id_to_uuid = {}
    
    # 1. Add new UUID columns to tables
    print("Adding UUID columns...")
    
    # Add UUID column to users
    session.execute(text("""
        ALTER TABLE users ADD COLUMN id_new UUID DEFAULT uuid_generate_v4() NOT NULL
    """))
    
    # Add UUID column to conversations
    session.execute(text("""
        ALTER TABLE conversations ADD COLUMN id_new UUID DEFAULT uuid_generate_v4() NOT NULL
    """))
    
    # Add UUID column to messages
    session.execute(text("""
        ALTER TABLE messages ADD COLUMN id_new UUID DEFAULT uuid_generate_v4() NOT NULL
    """))
    
    # Add UUID column to emotion_logs
    session.execute(text("""
        ALTER TABLE emotion_logs ADD COLUMN id_new UUID DEFAULT uuid_generate_v4() NOT NULL
    """))
    
    # Add new foreign key columns
    session.execute(text("""
        ALTER TABLE conversations ADD COLUMN user_id_new UUID REFERENCES users(id_new)
    """))
    
    session.execute(text("""
        ALTER TABLE messages ADD COLUMN conversation_id_new UUID REFERENCES conversations(id_new)
    """))
    
    session.execute(text("""
        ALTER TABLE emotion_logs ADD COLUMN user_id_new UUID REFERENCES users(id_new)
    """))
    
    session.execute(text("""
        ALTER TABLE emotion_logs ADD COLUMN conversation_id_new UUID REFERENCES conversations(id_new)
    """))
    
    session.commit()
    
    # 2. Populate new UUID columns and foreign key references
    print("Populating UUID columns...")
    
    # Get all users and map old IDs to new UUIDs
    users = session.execute(text("SELECT id FROM users")).fetchall()
    for user in users:
        old_id = user[0]
        new_uuid = str(uuid.uuid4())
        old_id_to_uuid[old_id] = new_uuid
        session.execute(text("UPDATE users SET id_new = :new_uuid WHERE id = :old_id"), {
            'new_uuid': new_uuid,
            'old_id': old_id
        })
    
    # Update conversations
    conversations = session.execute(text("SELECT id, user_id FROM conversations")).fetchall()
    for conv in conversations:
        old_id = conv[0]
        old_user_id = conv[1]
        new_uuid = str(uuid.uuid4())
        user_uuid = old_id_to_uuid.get(old_user_id)
        
        if user_uuid:
            conv_old_id_to_uuid[old_id] = new_uuid
            session.execute(text("""
                UPDATE conversations 
                SET id_new = :new_uuid, user_id_new = :user_uuid 
                WHERE id = :old_id
            """), {
                'new_uuid': new_uuid,
                'user_uuid': user_uuid,
                'old_id': old_id
            })
    
    # Update messages
    messages = session.execute(text("SELECT id, conversation_id FROM messages")).fetchall()
    for msg in messages:
        old_id = msg[0]
        old_conv_id = msg[1]
        new_uuid = str(uuid.uuid4())
        conv_uuid = conv_old_id_to_uuid.get(old_conv_id)
        
        if conv_uuid:
            session.execute(text("""
                UPDATE messages 
                SET id_new = :new_uuid, conversation_id_new = :conv_uuid 
                WHERE id = :old_id
            """), {
                'new_uuid': new_uuid,
                'conv_uuid': conv_uuid,
                'old_id': old_id
            })
    
    # Update emotion logs
    emotion_logs = session.execute(text("SELECT id, user_id, conversation_id FROM emotion_logs")).fetchall()
    for log in emotion_logs:
        old_id = log[0]
        old_user_id = log[1]
        old_conv_id = log[2]
        new_uuid = str(uuid.uuid4())
        user_uuid = old_id_to_uuid.get(old_user_id)
        conv_uuid = conv_old_id_to_uuid.get(old_conv_id)
        
        if user_uuid and conv_uuid:
            session.execute(text("""
                UPDATE emotion_logs 
                SET id_new = :new_uuid, user_id_new = :user_uuid, conversation_id_new = :conv_uuid 
                WHERE id = :old_id
            """), {
                'new_uuid': new_uuid,
                'user_uuid': user_uuid,
                'conv_uuid': conv_uuid,
                'old_id': old_id
            })
    
    session.commit()
    
    # 3. Drop old constraints and columns
    print("Dropping old constraints and columns...")
    
    # Drop foreign key constraints
    session.execute(text("ALTER TABLE conversations DROP CONSTRAINT conversations_user_id_fkey"))
    session.execute(text("ALTER TABLE messages DROP CONSTRAINT messages_conversation_id_fkey"))
    session.execute(text("ALTER TABLE emotion_logs DROP CONSTRAINT emotion_logs_user_id_fkey"))
    session.execute(text("ALTER TABLE emotion_logs DROP CONSTRAINT emotion_logs_conversation_id_fkey"))
    
    # Drop old primary key constraints
    session.execute(text("ALTER TABLE users DROP CONSTRAINT users_pkey"))
    session.execute(text("ALTER TABLE conversations DROP CONSTRAINT conversations_pkey"))
    session.execute(text("ALTER TABLE messages DROP CONSTRAINT messages_pkey"))
    session.execute(text("ALTER TABLE emotion_logs DROP CONSTRAINT emotion_logs_pkey"))
    
    session.commit()
    
    # 4. Rename columns to replace old ones
    print("Renaming columns...")
    
    # Rename UUID columns to replace old ID columns
    session.execute(text("ALTER TABLE users RENAME COLUMN id_new TO id"))
    session.execute(text("ALTER TABLE conversations RENAME COLUMN id_new TO id"))
    session.execute(text("ALTER TABLE messages RENAME COLUMN id_new TO id"))
    session.execute(text("ALTER TABLE emotion_logs RENAME COLUMN id_new TO id"))
    
    # Rename foreign key columns
    session.execute(text("ALTER TABLE conversations RENAME COLUMN user_id_new TO user_id"))
    session.execute(text("ALTER TABLE messages RENAME COLUMN conversation_id_new TO conversation_id"))
    session.execute(text("ALTER TABLE emotion_logs RENAME COLUMN user_id_new TO user_id"))
    session.execute(text("ALTER TABLE emotion_logs RENAME COLUMN conversation_id_new TO conversation_id"))
    
    session.commit()
    
    # 5. Recreate constraints
    print("Recreating constraints...")
    
    # Recreate primary key constraints
    session.execute(text("ALTER TABLE users ADD PRIMARY KEY (id)"))
    session.execute(text("ALTER TABLE conversations ADD PRIMARY KEY (id)"))
    session.execute(text("ALTER TABLE messages ADD PRIMARY KEY (id)"))
    session.execute(text("ALTER TABLE emotion_logs ADD PRIMARY KEY (id)"))
    
    # Recreate foreign key constraints
    session.execute(text("""
        ALTER TABLE conversations 
        ADD CONSTRAINT conversations_user_id_fkey 
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    """))
    
    session.execute(text("""
        ALTER TABLE messages 
        ADD CONSTRAINT messages_conversation_id_fkey 
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    """))
    
    session.execute(text("""
        ALTER TABLE emotion_logs 
        ADD CONSTRAINT emotion_logs_user_id_fkey 
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    """))
    
    session.execute(text("""
        ALTER TABLE emotion_logs 
        ADD CONSTRAINT emotion_logs_conversation_id_fkey 
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    """))
    
    session.commit()
    print("✅ PostgreSQL migration completed successfully")


if __name__ == "__main__":
    print("⚠️  WARNING: This will migrate your database from integer IDs to UUIDs.")
    print("⚠️  Please make sure you have a backup of your database before proceeding.")
    
    response = input("Do you want to continue? (yes/no): ")
    if response.lower() == 'yes':
        run_migration()
    else:
        print("Migration cancelled.")