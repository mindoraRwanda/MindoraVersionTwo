import asyncio
import time
import uuid
import sys
import os
import json
import matplotlib.pyplot as plt
from sqlalchemy.orm import Session

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import llm_providers FIRST to avoid Pydantic V1/V2 conflicts
from app.services.llm_providers import create_llm_provider
import app.services.llm_providers

from app.db.database import SessionLocal, engine
from app.db.models import User, Conversation, Message

from app.services.stateful_pipeline import initialize_stateful_pipeline

# Ensure tables exist (should already be there, but good for safety)
from app.db import models
models.Base.metadata.create_all(bind=engine)

async def measure_latency():
    print("🚀 Starting LLM Pipeline Latency Measurement...")
    
    # Initialize DB session
    db = SessionLocal()
    
    # Create Test User and Conversation
    user_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    
    test_user_email = f"latency_test_{user_id.hex[:8]}@example.com"
    
    try:
        # Create dummy user
        test_user = User(
            id=user_id,
            uuid=user_id,
            username=f"latency_tester_{user_id.hex[:8]}",
            email=test_user_email,
            password="testpassword", # Correct field name
            gender="female"
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        print(f"👤 Created test user: {test_user.email}")

        conversation = Conversation(
            id=conversation_id,
            uuid=conversation_id,
            user_id=test_user.id,
            # title="Latency Test Conversation" # title might not exist in model, check later if needed
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        print(f"💬 Created test conversation: {conversation.uuid}")

        # Initialize Pipeline
        print("🔧 Initializing LLM Provider and Pipeline...")
        # Using the model requested by user in previous steps
        llm_provider = create_llm_provider(provider="ollama", model="granite4:1b-h")
        pipeline = initialize_stateful_pipeline(llm_provider=llm_provider, db=db)
        print("✅ Pipeline initialized")

        # Load Realistic Queries
        queries_file = os.path.join(os.path.dirname(__file__), "realistic_queries.json")
        print(f"📂 Loading queries from {queries_file}...")
        with open(queries_file, "r") as f:
            queries = json.load(f)[:3] # Limit to 3 queries for faster testing
        print(f"✅ Loaded {len(queries)} queries (limited for testing)")

        print("\n⏱️  Beginning Latency Measurements...")
        print("-" * 60)
        print(f"{'Query':<40} | {'Latency (s)':<15}")
        print("-" * 60)

        latencies = []
        processed_queries = []
        
        for i, query in enumerate(queries):
            # Create user message record (simulating router behavior)
            user_msg = Message(
                conversation_id=conversation.id,
                sender="user",
                content=query
            )
            db.add(user_msg)
            db.commit()
            db.refresh(user_msg)

            # Get history
            recent_history = db.query(Message)\
                .filter_by(conversation_id=conversation.id)\
                .order_by(Message.timestamp.desc())\
                .limit(15)\
                .all()
            recent_history.reverse()
            conversation_history = [
                {"role": msg.sender, "text": msg.content} for msg in recent_history
            ]

            # Measure Pipeline Execution
            start_time = time.time()
            
            result = await pipeline.process_query(
                query=query,
                user_id=str(test_user.id),
                conversation_id=str(conversation.id),
                message_id=str(user_msg.id),
                conversation_history=conversation_history,
                user_gender=str(test_user.gender),
                db=db
            )
            
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            processed_queries.append(query)

            # Save bot response (to keep history consistent for next turn)
            bot_reply = result.get("response", "")
            bot_msg = Message(
                conversation_id=conversation.id,
                sender="bot",
                content=bot_reply
            )
            db.add(bot_msg)
            db.commit()

            print(f"{query[:37]+'...' if len(query)>37 else query:<40} | {latency:.4f} s")

        print("-" * 60)
        avg_latency = sum(latencies) / len(latencies)
        print(f"📊 Average Latency: {avg_latency:.4f} s")
        print("-" * 60)

        # Generate Plot
        print("📊 Generating latency plot...")
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(latencies) + 1), latencies, marker='o', linestyle='-', color='b')
        plt.title('LLM Pipeline Latency per Query')
        plt.xlabel('Query Number')
        plt.ylabel('Latency (seconds)')
        plt.grid(True)
        plt.xticks(range(1, len(latencies) + 1))
        
        # Add average line
        plt.axhline(y=avg_latency, color='r', linestyle='--', label=f'Average: {avg_latency:.2f}s')
        plt.legend()
        
        plot_path = os.path.join(os.path.dirname(__file__), "latency_plot.png")
        plt.savefig(plot_path)
        print(f"✅ Plot saved to {plot_path}")

    except Exception as e:
        print(f"❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\n🧹 Cleaning up test data...")
        try:
            # Delete messages first due to FK (cascade might handle it but explicit is safer)
            if 'conversation' in locals():
                db.query(Message).filter(Message.conversation_id == conversation.id).delete()
                db.delete(conversation)
            if 'test_user' in locals():
                db.delete(test_user)
            db.commit()
            print("✅ Cleanup complete")
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")
        finally:
            db.close()

if __name__ == "__main__":
    asyncio.run(measure_latency())
