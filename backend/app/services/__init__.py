"""
Services package for the Mindora mental health chatbot.

This package provides all the core services for the chatbot including:
- LLM provider for model access
- Stateful pipeline for message processing
- Session management for conversation state
- Crisis detection and intervention
- Query validation and processing
- Cultural context integration
- Safety and content filtering

Usage:
    from backend.app.services import service_container, get_service

    # Initialize all services
    await service_container.initialize_all_services()

    # Get a specific service
    stateful_pipeline = get_service("stateful_pipeline")
    session_manager = get_service("session_manager")

    # Use the services
    result = await stateful_pipeline.process_query(query, user_id, conversation_history)

    # Cleanup when done
    await service_container.shutdown_all_services()
"""

from .service_container import (
    ServiceContainer,
    ServiceRegistry,
    ServiceConfig,
    service_container,
    get_service,
    initialize_services,
    shutdown_services,
    service_lifecycle,
    check_service_health
)

__all__ = [
    "ServiceContainer",
    "ServiceRegistry",
    "ServiceConfig",
    "service_container",
    "get_service",
    "initialize_services",
    "shutdown_services",
    "service_lifecycle",
    "check_service_health"
]