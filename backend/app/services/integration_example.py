"""
Example integration of the service container with FastAPI.

This module demonstrates how to integrate the global service container
into the main FastAPI application for proper service lifecycle management.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

from fastapi import FastAPI, Depends
from backend.app.services.service_container import (
    service_container,
    check_service_health,
    ServiceContainer
)


# Dependency injection functions
def get_llm_service() -> Any:
    """Dependency injection for LLM service."""
    return service_container.get_service("llm_service")


def get_session_manager() -> Any:
    """Dependency injection for session manager."""
    return service_container.get_service("session_manager")


def get_state_router() -> Any:
    """Dependency injection for state router."""
    return service_container.get_service("langgraph_state_router")


def get_query_validator() -> Any:
    """Dependency injection for query validator."""
    return service_container.get_service("query_validator")


def get_crisis_interceptor() -> Any:
    """Dependency injection for crisis interceptor."""
    return service_container.get_service("crisis_interceptor")


# Lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager for service initialization and cleanup.

    This ensures all services are properly initialized when the application starts
    and cleaned up when the application shuts down.
    """
    print("🚀 Starting Mindora chatbot application...")

    # Initialize all services
    success = await service_container.initialize_all_services()
    if not success:
        print("❌ Failed to initialize services")
        raise RuntimeError("Service initialization failed")

    # Check service health
    health_status = await check_service_health()
    print("📊 Service Health Status:")
    for service_name, status in health_status.items():
        status_icon = "✅" if status["healthy"] else "❌"
        print(f"  {status_icon} {service_name}")

    print("✅ Application startup complete")

    yield  # Application runs here

    # Cleanup
    print("🛑 Shutting down application...")
    await service_container.shutdown_all_services()
    print("✅ Application shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application with service integration.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Mindora Chatbot API",
        description="Mental Health Support Chatbot with Rwandan Cultural Context",
        version="1.0.0",
        lifespan=lifespan
    )

    return app


# Example usage in route handlers
async def example_chat_endpoint(
    message: str,
    session_id: str,
    llm_service = Depends(get_llm_service),
    session_manager = Depends(get_session_manager),
    state_router = Depends(get_state_router)
):
    """
    Example chat endpoint showing how to use the services.

    This demonstrates the proper way to use dependency injection
    to access the initialized services.
    """
    try:
        # Use the services
        response = await state_router.route_conversation(
            session_id=session_id,
            user_message=message
        )

        return {
            "response": response["response"],
            "state": response["next_state"],
            "confidence": response["confidence"]
        }

    except Exception as e:
        # Handle errors gracefully
        return {
            "error": "An error occurred while processing your message",
            "details": str(e)
        }


# Example of manual service access (alternative to dependency injection)
async def example_manual_service_access():
    """Example of accessing services manually."""

    # Get services directly from the container
    llm_service = service_container.get_service("llm_service")
    session_manager = service_container.get_service("session_manager")

    # Use the services
    if llm_service and session_manager:
        # Your service logic here
        pass

    return {"status": "success"}


# Health check endpoint
async def health_check():
    """Health check endpoint for monitoring."""
    health_status = await check_service_health()

    # Count healthy vs unhealthy services
    healthy_count = sum(1 for status in health_status.values() if status["healthy"])
    total_count = len(health_status)

    return {
        "status": "healthy" if healthy_count == total_count else "degraded",
        "healthy_services": healthy_count,
        "total_services": total_count,
        "services": health_status
    }


# Example of how to add routes to the FastAPI app
def setup_routes(app: FastAPI):
    """Set up API routes with service integration."""

    @app.post("/chat")
    async def chat_endpoint(
        message: str,
        session_id: str = "default",
        llm_service = Depends(get_llm_service),
        session_manager = Depends(get_session_manager),
        state_router = Depends(get_state_router)
    ):
        """Chat endpoint with service integration."""
        return await example_chat_endpoint(
            message, session_id, llm_service, session_manager, state_router
        )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return await health_check()

    @app.post("/services/initialize")
    async def initialize_services_endpoint():
        """Manual service initialization endpoint."""
        success = await service_container.initialize_all_services()
        return {"success": success}

    @app.post("/services/shutdown")
    async def shutdown_services_endpoint():
        """Manual service shutdown endpoint."""
        await service_container.shutdown_all_services()
        return {"success": True}


# Convenience function to run the application
def run_app():
    """Run the FastAPI application with service integration."""
    app = create_app()
    setup_routes(app)

    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    run_app()