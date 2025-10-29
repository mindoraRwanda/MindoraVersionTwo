import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Routers
from backend.app.routers.chat_router import router as chat_router
from backend.app.auth.auth_router import router as auth_router
from backend.app.auth.emotion_router import router as emotion_router
from backend.app.auth.emotion_router import router as mental_health_router
from backend.app.routers.conversations_router import router as conversations_router
from backend.app.routers.messages_router import router as messages_router

#voice
from backend.app.routers.voice_router import router as voice_router  # <-- add



# DB
from backend.app.db.database import engine
from backend.app.db import models
from backend.app.db.database import Base

# Services
from backend.app.services.llm_service_refactored import LLMService
from backend.app.services.emotion_classifier import initialize_emotion_classifier
from backend.app.services.rag_service import initialize_rag_service
from backend.app.services.query_validator_langgraph import initialize_langgraph_query_validator

# Global service instances
llm_service = None
emotion_classifier = None
rag_service = None
query_validator = None

# Create tables if missing (non-destructive)
models.Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global llm_service, emotion_classifier, rag_service, query_validator

    print("🚀 Starting up Therapy Chatbot API...")

    # 1) Emotion Classifier
    try:
        emotion_classifier = initialize_emotion_classifier()
        print("✅ Emotion classifier initialized successfully")
    except Exception as e:
        print(f"⚠️  Warning: Emotion classifier failed to initialize: {e}")

    # 2) RAG bootstrap ONLY when explicitly requested
    try:
        if os.getenv("RAG_BOOTSTRAP") == "1":
            print("📥 RAG_BOOTSTRAP=1 detected — starting one-time ingestion…")
            rag_service = initialize_rag_service()
            print("✅ RAG bootstrap (ingestion) completed")
        else:
            print("⏭️  Skipping RAG bootstrap; expecting existing Qdrant collection")
    except Exception as e:
        print(f"⚠️  Warning: RAG bootstrap failed: {e}")

    # 3) LLM Service
    llm_service = LLMService(use_vllm=False, provider_name="groq", model_name="openai/gpt-oss-120b")
    if not llm_service.initialize():
        print("⚠️  Warning: LLM service failed to initialize. Some features may not work.")
        print(f"Error: {llm_service.initialization_error}")
    else:
        print("✅ LLM service initialized successfully")

    # 4) LangGraph Query Validator
    try:
        query_validator = initialize_langgraph_query_validator(llm_service.llm_provider if llm_service else None)
        print("✅ LangGraph query validator initialized successfully")
    except Exception as e:
        print(f"⚠️  Warning: LangGraph query validator failed to initialize: {e}")

    yield
    print("🛑 Shutting down Therapy Chatbot API...")


app = FastAPI(
    title="Therapy Chatbot API",
    lifespan=lifespan
)

# Routers
app.include_router(auth_router)
app.include_router(conversations_router)
app.include_router(messages_router)
app.include_router(chat_router)
app.include_router(emotion_router)
app.include_router(mental_health_router, prefix="/insights")
#voice
app.include_router(voice_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Therapy Chatbot API is running"}


# second old code
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from contextlib import asynccontextmanager
# from backend.app.routers.chat_router import router as chat_router

# # Import your database engine and models
# from backend.app.db.database import engine
# from backend.app.db import models
# from backend.app.db.database import Base
# from backend.app.auth.auth_router import router as auth_router
# from backend.app.auth.emotion_router import router as emotion_router
# from backend.app.auth.emotion_router import router as mental_health_router
# from backend.app.routers.conversations_router import router as conversations_router
# from backend.app.routers.messages_router import router as messages_router

# # Import the refactored LLM service
# from backend.app.services.llm_service_refactored import LLMService
# from backend.app.services.emotion_classifier import initialize_emotion_classifier
# from backend.app.services.rag_service import initialize_rag_service
# from backend.app.services.query_validator_langgraph import initialize_langgraph_query_validator

# # Global service instances
# llm_service = None
# emotion_classifier = None
# rag_service = None
# query_validator = None

# # TEMPORARY: Drop everything before recreating
# models.Base.metadata.create_all(bind=engine)  # ❗️ This rebuilds all tables


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Application lifespan manager for startup and shutdown events."""
#     global llm_service, emotion_classifier, rag_service, query_validator

#     # Startup: Initialize all services
#     print("🚀 Starting up Therapy Chatbot API...")
    
#     # Initialize Emotion Classifier
#     try:
#         emotion_classifier = initialize_emotion_classifier()
#         print("✅ Emotion classifier initialized successfully")
#     except Exception as e:
#         print(f"⚠️  Warning: Emotion classifier failed to initialize: {e}")

#     # Initialize RAG Service
#     try:
#         rag_service = initialize_rag_service()
#         print("✅ RAG service initialized successfully")
#     except Exception as e:
#         print(f"⚠️  Warning: RAG service failed to initialize: {e}")

#     # Initialize LLM Service
#     llm_service = LLMService(use_vllm=False, provider_name="groq", model_name="openai/gpt-oss-120b")  # Set to True if using vLLM
#     if not llm_service.initialize():
#         print("⚠️  Warning: LLM service failed to initialize. Some features may not work.")
#         print(f"Error: {llm_service.initialization_error}")
#     else:
#         print("✅ LLM service initialized successfully")

#     # Initialize LangGraph Query Validator
#     try:
#         query_validator = initialize_langgraph_query_validator(llm_service.llm_provider if llm_service else None)
#         print("✅ LangGraph query validator initialized successfully")
#     except Exception as e:
#         print(f"⚠️  Warning: LangGraph query validator failed to initialize: {e}")

#     yield

#     # Shutdown: Cleanup resources
#     print("🛑 Shutting down Therapy Chatbot API...")


# app = FastAPI(
#     title="Therapy Chatbot API",
#     lifespan=lifespan
# )

# # Include routers
# app.include_router(auth_router)  # Authentication endpoints
# app.include_router(conversations_router)  # Conversation management
# app.include_router(messages_router)  # Message handling
# app.include_router(chat_router)  # Chat functionality

# # CORS for frontend (adjust domains as needed)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Frontend origin
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# async def root():
#     return {"message": "Therapy Chatbot API is running"}

# app.include_router(emotion_router)
# app.include_router(mental_health_router, prefix="/insights")




# First Old Code

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



#New code
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# from backend.app.routers.chat_router import router as chat_router
# from backend.app.auth.auth_router import router as auth_router
# from backend.app.auth.emotion_router import router as emotion_router
# from backend.app.auth.emotion_router import router as mental_health_router  # ✅ Correct source

# from backend.app.db.database import engine
# from backend.app.db import models

# # Create DB tables
# models.Base.metadata.create_all(bind=engine)

# app = FastAPI(title="Therapy Chatbot API")

# # CORS setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # adjust as needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Register routers
# app.include_router(auth_router)
# app.include_router(chat_router)
# app.include_router(emotion_router)
# app.include_router(mental_health_router, prefix="/insights")  # 

# @app.get("/")
# async def root():
#     return {"message": "Therapy Chatbot API is running"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
