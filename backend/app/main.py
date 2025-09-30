import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routers.chat_router import router as chat_router

# Import your database engine and models
from backend.app.db.database import engine
from backend.app.db import models
from backend.app.auth.auth_router import router as auth_router
from backend.app.auth.emotion_router import router as emotion_router
from backend.app.auth.emotion_router import router as mental_health_router
from backend.app.routers.conversations_router import router as conversations_router
from backend.app.routers.messages_router import router as messages_router

# Import service container for global service management
from backend.app.services.service_container import service_container, check_service_health

# TEMPORARY: Drop everything before recreating
models.Base.metadata.create_all(bind=engine)  # ❗️ This rebuilds all tables

# Create FastAPI app first
app = FastAPI(
    title="Therapy Chatbot API"
)

# Startup event handler for service initialization
@app.on_event("startup")
async def startup_event():
    """Initialize services during application startup."""
    print("🚀 Starting up Therapy Chatbot API with service container...")

    try:
        # Initialize all services using the service container
        print("🔧 Initializing services...")
        success = await service_container.initialize_all_services()
        if not success:
            print("❌ Failed to initialize services")
            raise RuntimeError("Service initialization failed")

        # Check service health
        print("🏥 Checking service health...")
        health_status = await check_service_health()
        print("📊 Service Health Status:")
        healthy_count = 0
        for service_name, status in health_status.items():
            status_icon = "✅" if status["healthy"] else "❌"
            print(f"  {status_icon} {service_name}")
            if status["healthy"]:
                healthy_count += 1

        print(f"✅ {healthy_count}/{len(health_status)} services healthy")
        print("✅ Application startup complete")

    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise

# Shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services during application shutdown."""
    print("🛑 Shutting down Therapy Chatbot API...")
    await service_container.shutdown_all_services()
    print("✅ Application shutdown complete")

# Include routers
app.include_router(auth_router)  # Authentication endpoints
app.include_router(conversations_router)  # Conversation management
app.include_router(messages_router)  # Message handling
app.include_router(chat_router)  # Chat functionality

# CORS for frontend (adjust domains as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Therapy Chatbot API is running"}

app.include_router(emotion_router)
app.include_router(mental_health_router, prefix="/insights")
