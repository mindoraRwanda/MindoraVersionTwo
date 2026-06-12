import os
from dotenv import load_dotenv
from pathlib import Path

# Get the environment variable
environment = os.getenv("ENVIRONMENT", "development").lower()

# Construct the path to the environment-specific .env file in the root directory
root_dir = Path(__file__).parent.parent.parent  # Go up three levels from backend/app/main.py
env_file = root_dir / f".env.{environment}"

# Load environment variables: environment-specific file → .env → .env.example
print(f"🔧 ENVIRONMENT={environment!r}, looking for {env_file}")
if env_file.exists():
    load_dotenv(env_file, override=True)
    print(f"🔧 Loaded environment variables from {env_file}")
elif (root_dir / ".env").exists():
    load_dotenv(root_dir / ".env", override=True)
    print(f"🔧 Loaded environment variables from {root_dir / '.env'}")
elif (root_dir / ".env.example").exists():
    load_dotenv(root_dir / ".env.example", override=True)
    print(f"⚠️ Loaded environment variables from .env.example (placeholder values — create a .env file)")
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

    # Initialize all services — never raise here so the app always starts.
    # If a service fails, the pipeline nodes will return clear error messages
    # rather than crashing the whole process.
    print("🔧 Initializing services...")
    success = await service_container.initialize_all_services()

    # Always run a health check so the logs tell us exactly what failed.
    print("🏥 Service Health Status:")
    try:
        health_status = await check_service_health()
        healthy_count = 0
        for service_name, status in health_status.items():
            status_icon = "✅" if status.get("healthy") else "❌"
            error_detail = f" — {status['error']}" if status.get("error") else ""
            print(f"  {status_icon} {service_name}{error_detail}")
            if status.get("healthy"):
                healthy_count += 1
        print(f"{'✅' if success else '⚠️'} {healthy_count}/{len(health_status)} services healthy")
    except Exception as e:
        print(f"⚠️ Health check failed: {e}")

    if not success:
        print("⚠️ Some services failed — app running in degraded mode. Check logs above for details.")
    else:
        print("✅ Application startup complete")

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
