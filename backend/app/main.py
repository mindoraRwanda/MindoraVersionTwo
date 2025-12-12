import os
from dotenv import load_dotenv
from pathlib import Path

# Get the environment variable
environment = os.getenv("ENVIRONMENT", "development").lower()

# Construct the path to the environment-specific .env file in the root directory
root_dir = Path(__file__).parent.parent.parent  # Go up three levels from backend/app/main.py
env_file = root_dir / f".env.{environment}"

# Load environment variables from the environment-specific file
# Fall back to .env.example if the specific file doesn't exist
if env_file.exists():
    load_dotenv(env_file, override=True)
    print(f"🔧 Loaded environment variables from {env_file}")
else:
    # Try to load from .env.example as a fallback
    example_env_file = root_dir / ".env.example"
    if example_env_file.exists():
        load_dotenv(example_env_file, override=True)
        print(f"🔧 Loaded environment variables from {example_env_file} (fallback)")
    else:
        print("⚠️ No environment file found, using system environment variables only")

# Initialize the new settings system (lazy-loaded)
from .settings import settings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Import your database engine and models
try:
    from .db.database import engine
    from .db import models
except ImportError:
    from .db.database import engine
    from .db import models

try:
    from .auth.auth_router import router as auth_router
    from .auth.emotion_router import router as emotion_router
    from .auth.emotion_router import router as mental_health_router
except ImportError:
    from .auth.auth_router import router as auth_router
    from .auth.emotion_router import router as emotion_router
    from .auth.emotion_router import router as mental_health_router

try:
    from .routers.conversations_router import router as conversations_router
    from .routers.messages_router import router as messages_router
    from .routers.voice_router import router as voice_router
except ImportError:
    from .routers.conversations_router import router as conversations_router
    from .routers.messages_router import router as messages_router
    from .routers.voice_router import router as voice_router

# Import service container for global service management
try:
    from .services.service_container import service_container, check_service_health
except ImportError:
    from .services.service_container import service_container, check_service_health

# Initialize KB (cards) on startup, similar to reference app
try:
    from .services.kb import initialize_kb
except ImportError:
    initialize_kb = None

# TEMPORARY: Drop everything before recreating
models.Base.metadata.create_all(bind=engine)  # ❗️ This rebuilds all tables

# Create FastAPI app first
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    debug=settings.debug
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

        # Initialize KB cards-based retrieval
        if initialize_kb is not None:
            print("📚 Initializing KB cards...")
            initialize_kb()

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
app.include_router(voice_router)  # Voice message handling

# CORS for frontend (using settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # From settings
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Therapy Chatbot API is running"}

app.include_router(emotion_router)
app.include_router(mental_health_router, prefix="/insights")
