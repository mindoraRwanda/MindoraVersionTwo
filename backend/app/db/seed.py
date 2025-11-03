#!/usr/bin/env python3
"""
Database seeder script for populating the Mindora database with sample data.
This script creates realistic test data including users, therapists, and their relationships.
"""

import sys
import os
from datetime import datetime, timedelta
import random
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

# Add the backend directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .database import SessionLocal, engine, Base
from .models import (
    User, Therapist, UserTherapist, Conversation, Message, EmotionLog, CrisisLog,
    SenderType, CrisisType, CrisisSeverity, CrisisStatus
)

# Sample data
THERAPISTS_DATA = [
    {
        "full_name": "Dr. Sarah Johnson",
        "email": "sarah.johnson@mindora.com",
        "phone": "+1-555-0101",
        "specialization": "Cognitive Behavioral Therapy",
        "gender": "female",
        "active": True
    },
    {
        "full_name": "Dr. Michael Chen",
        "email": "michael.chen@mindora.com",
        "phone": "+1-555-0102",
        "specialization": "Trauma-Focused Therapy",
        "gender": "male",
        "active": True
    },
    {
        "full_name": "Dr. Emily Rodriguez",
        "email": "emily.rodriguez@mindora.com",
        "phone": "+1-555-0103",
        "specialization": "Family Therapy",
        "gender": "female",
        "active": True
    },
    {
        "full_name": "Dr. James Wilson",
        "email": "james.wilson@mindora.com",
        "phone": "+1-555-0104",
        "specialization": "Anxiety Disorders",
        "gender": "male",
        "active": True
    },
    {
        "full_name": "Dr. Lisa Thompson",
        "email": "lisa.thompson@mindora.com",
        "phone": "+1-555-0105",
        "specialization": "Depression Treatment",
        "gender": "female",
        "active": True
    }
]

USERS_DATA = [
    {
        "username": "alice_smith",
        "phone": "+1-555-1001",
        "email": "alice.smith@example.com",
        "password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fWdXe9dO",  # "password123"
        "gender": "female"
    },
    {
        "username": "bob_jones",
        "phone": "+1-555-1002",
        "email": "bob.jones@example.com",
        "password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fWdXe9dO",
        "gender": "male"
    },
    {
        "username": "carol_davis",
        "phone": "+1-555-1003",
        "email": "carol.davis@example.com",
        "password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fWdXe9dO",
        "gender": "female"
    },
    {
        "username": "david_brown",
        "phone": "+1-555-1004",
        "email": "david.brown@example.com",
        "password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fWdXe9dO",
        "gender": "male"
    },
    {
        "username": "emma_wilson",
        "phone": "+1-555-1005",
        "email": "emma.wilson@example.com",
        "password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fWdXe9dO",
        "gender": "female"
    },
    {
        "username": "frank_miller",
        "phone": "+1-555-1006",
        "email": "frank.miller@example.com",
        "password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fWdXe9dO",
        "gender": "male"
    },
    {
        "username": "grace_lee",
        "phone": "+1-555-1007",
        "email": "grace.lee@example.com",
        "password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fWdXe9dO",
        "gender": "female"
    },
    {
        "username": "henry_taylor",
        "phone": "+1-555-1008",
        "email": "henry.taylor@example.com",
        "password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fWdXe9dO",
        "gender": "male"
    }
]

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]

SAMPLE_CONVERSATIONS = [
    {
        "messages": [
            {"sender": SenderType.user, "content": "Hi, I've been feeling really anxious lately."},
            {"sender": SenderType.bot, "content": "I'm sorry to hear that. Can you tell me more about what's been causing your anxiety?"},
            {"sender": SenderType.user, "content": "Work has been really stressful, and I can't seem to relax."},
            {"sender": SenderType.bot, "content": "That sounds challenging. Let's explore some coping strategies together."}
        ],
        "emotions": ["fear", "sadness", "fear", "trust"]
    },
    {
        "messages": [
            {"sender": SenderType.user, "content": "I had a great day today! Everything went well."},
            {"sender": SenderType.bot, "content": "That's wonderful! What made your day so positive?"},
            {"sender": SenderType.user, "content": "I finished a big project and got praised by my boss."},
            {"sender": SenderType.bot, "content": "Congratulations! It's great that you're recognizing your achievements."}
        ],
        "emotions": ["joy", "joy", "joy", "joy"]
    },
    {
        "messages": [
            {"sender": SenderType.user, "content": "I'm really struggling with motivation. I just want to stay in bed all day."},
            {"sender": SenderType.bot, "content": "I understand that feeling. Sometimes motivation can be hard to find. What usually helps you get started?"},
            {"sender": SenderType.user, "content": "I don't know. I feel so tired all the time."},
            {"sender": SenderType.bot, "content": "Fatigue can be a sign of various things. Let's talk about your sleep and energy levels."}
        ],
        "emotions": ["sadness", "trust", "sadness", "trust"]
    }
]

def seed_therapists(db: Session):
    """Seed therapist data."""
    print("Seeding therapists...")
    for therapist_data in THERAPISTS_DATA:
        therapist = Therapist(**therapist_data)
        try:
            db.add(therapist)
            db.commit()
            print(f"Added therapist: {therapist.full_name}")
        except IntegrityError:
            db.rollback()
            print(f"Therapist {therapist_data['email']} already exists, skipping...")

def seed_users(db: Session):
    """Seed user data."""
    print("Seeding users...")
    for user_data in USERS_DATA:
        user = User(**user_data)
        try:
            db.add(user)
            db.commit()
            print(f"Added user: {user.username}")
        except IntegrityError:
            db.rollback()
            print(f"User {user_data['username']} already exists, skipping...")

def seed_user_therapist_relationships(db: Session):
    """Establish relationships between users and therapists."""
    print("Seeding user-therapist relationships...")

    # Get all users and therapists
    users = db.query(User).all()
    therapists = db.query(Therapist).all()

    if not users or not therapists:
        print("No users or therapists found, skipping relationships...")
        return

    # Assign therapists to users (some users get multiple therapists)
    for i, user in enumerate(users):
        # Each user gets 1-2 therapists
        num_therapists = random.randint(1, 2)
        assigned_therapists = random.sample(therapists, num_therapists)

        for j, therapist in enumerate(assigned_therapists):
            relationship = UserTherapist(
                user_id=user.id,
                therapist_id=therapist.id,
                is_primary=(j == 0),  # First therapist is primary
                status="active"
            )
            try:
                db.add(relationship)
                db.commit()
                print(f"Assigned {therapist.full_name} to {user.username} (primary: {relationship.is_primary})")
            except IntegrityError:
                db.rollback()
                print(f"Relationship between {user.username} and {therapist.full_name} already exists, skipping...")

def seed_conversations_and_messages(db: Session):
    """Seed conversations and messages with emotion logs."""
    print("Seeding conversations and messages...")

    users = db.query(User).all()
    if not users:
        print("No users found, skipping conversations...")
        return

    for user in users:
        # Each user gets 1-3 conversations
        num_conversations = random.randint(1, 3)

        for conv_idx in range(num_conversations):
            # Create conversation
            conversation = Conversation(
                user_id=user.id,
                started_at=datetime.now() - timedelta(days=random.randint(1, 30)),
                meta={"topic": f"Conversation {conv_idx + 1}"}
            )
            db.add(conversation)
            db.commit()

            # Choose a sample conversation template
            sample_conv = random.choice(SAMPLE_CONVERSATIONS)

            # Add messages with timestamps
            base_time = conversation.started_at
            for msg_idx, msg_data in enumerate(sample_conv["messages"]):
                message = Message(
                    conversation_id=conversation.id,
                    sender=msg_data["sender"],
                    content=msg_data["content"],
                    timestamp=base_time + timedelta(minutes=msg_idx * 5),
                    meta={}
                )
                db.add(message)
                db.commit()

                # Add emotion log for user messages
                if msg_data["sender"] == SenderType.user:
                    emotion = sample_conv["emotions"][msg_idx]
                    emotion_log = EmotionLog(
                        user_id=user.id,
                        conversation_id=conversation.id,
                        input_text=msg_data["content"],
                        detected_emotion=emotion,
                        timestamp=message.timestamp
                    )
                    db.add(emotion_log)
                    db.commit()

            print(f"Added conversation with {len(sample_conv['messages'])} messages for {user.username}")

def seed_crisis_logs(db: Session):
    """Seed some crisis logs for testing."""
    print("Seeding crisis logs...")

    # Get some users and their conversations
    users_with_conversations = db.query(User).join(Conversation).all()
    if not users_with_conversations:
        print("No users with conversations found, skipping crisis logs...")
        return

    # Create a few crisis logs
    crisis_scenarios = [
        {
            "type": CrisisType.self_harm,
            "severity": CrisisSeverity.moderate,
            "excerpt": "I've been having thoughts of hurting myself",
            "confidence": 0.85
        },
        {
            "type": CrisisType.suicide_ideation,
            "severity": CrisisSeverity.high,
            "excerpt": "I don't want to live anymore",
            "confidence": 0.92
        },
        {
            "type": CrisisType.substance_abuse,
            "severity": CrisisSeverity.low,
            "excerpt": "I've been drinking too much lately",
            "confidence": 0.78
        }
    ]

    for scenario in crisis_scenarios:
        # Pick a random user with conversations
        user = random.choice(users_with_conversations)
        user_conversations = db.query(Conversation).filter(Conversation.user_id == user.id).all()
        conversation = random.choice(user_conversations) if user_conversations else None

        # Get a random message from the conversation
        if conversation:
            messages = db.query(Message).filter(
                Message.conversation_id == conversation.id,
                Message.sender == SenderType.user
            ).all()
            message = random.choice(messages) if messages else None
        else:
            message = None

        # Use raw SQL to avoid SQLAlchemy casting issues
        try:
            db.execute(
                text("""
                INSERT INTO crisis_logs (
                    id, user_id, conversation_id, message_id, detected_type, severity,
                    confidence, excerpt, rationale, classifier_model, classifier_version,
                    status, notified_therapist_id
                ) VALUES (
                    gen_random_uuid(), :user_id, :conversation_id, :message_id, :detected_type,
                    :severity, :confidence, :excerpt, :rationale, :classifier_model,
                    :classifier_version, :status, :notified_therapist_id
                )
                """),
                {
                    "user_id": user.id,
                    "conversation_id": conversation.id if conversation else None,
                    "message_id": message.id if message else None,
                    "detected_type": scenario["type"].value,
                    "severity": scenario["severity"].value,
                    "confidence": scenario["confidence"],
                    "excerpt": scenario["excerpt"],
                    "rationale": "Detected through keyword analysis and context evaluation",
                    "classifier_model": "llama3-8b-8192",
                    "classifier_version": "v1.0",
                    "status": CrisisStatus.new.value,
                    "notified_therapist_id": None
                }
            )
            db.commit()
            print(f"Added crisis log for {user.username}: {scenario['type'].value}")
        except Exception as e:
            db.rollback()
            print(f"Error adding crisis log for {user.username}: {e}")

def main():
    """Main seeding function."""
    print("Starting database seeding...")

    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        seed_therapists(db)
        seed_users(db)
        seed_user_therapist_relationships(db)
        seed_conversations_and_messages(db)
        seed_crisis_logs(db)

        print("\nSeeding completed successfully!")

    except Exception as e:
        print(f"Error during seeding: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()