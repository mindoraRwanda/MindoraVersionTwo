"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.database import init_db
from app.routers import auth, conversations, messages, notifications, core
from app.services.kb import initialize_kb

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Mindora Mental Health Assistant API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
cors_origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(conversations.router)
app.include_router(messages.router)
app.include_router(notifications.router)
app.include_router(core.router)


@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup."""
    init_db()
    initialize_kb()


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Mindora API",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }

