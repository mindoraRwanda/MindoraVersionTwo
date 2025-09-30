# Service Container Documentation

This document explains how to use the global service container for centralized service initialization and management in the Mindora chatbot application.

## Overview

The service container provides a centralized way to:
- Initialize all services in the correct order with proper dependency injection
- Manage service lifecycle (startup and shutdown)
- Access services throughout the application
- Handle service health monitoring
- Provide graceful error handling for missing services

## Quick Start

### Basic Usage

```python
import asyncio
from backend.app.services import service_container, get_service

async def main():
    # Initialize all services
    success = await service_container.initialize_all_services()
    if not success:
        print("Failed to initialize services")
        return

    # Get a service
    llm_service = get_service("llm_service")
    session_manager = get_service("session_manager")

    # Use the services
    # ... your application logic here

    # Cleanup
    await service_container.shutdown_all_services()

# Run the application
asyncio.run(main())
```

### Using the Service Lifecycle Context Manager

```python
from backend.app.services import service_lifecycle

async def main():
    async with service_lifecycle() as services:
        # All services are initialized here
        llm_service = services.get_service("llm_service")

        # Your application logic here
        # Services are automatically cleaned up when exiting the context
```

### Integration with FastAPI

```python
from fastapi import FastAPI, Depends
from backend.app.services.integration_example import create_app

# Create app with automatic service management
app = create_app()

# Use dependency injection in your routes
@app.post("/chat")
async def chat(
    message: str,
    llm_service = Depends(get_llm_service),
    session_manager = Depends(get_session_manager)
):
    # Use the services
    return {"response": "Hello!"}
```

## Available Services

The following services are automatically registered and initialized:

### Core Services
- **llm_config**: LLM configuration management
- **database**: Database operations
- **session_manager**: Conversation session management
- **llm_service**: Main LLM service for response generation

### Processing Services
- **query_validator**: Query validation and classification
- **crisis_interceptor**: Crisis detection and intervention
- **emotion_classifier**: Emotion detection and analysis
- **state_router**: Basic state routing (FSM)
- **langgraph_state_router**: Advanced LLM-enhanced state routing

### Enhancement Services
- **retriever_service**: Vector retrieval for RAG
- **rag_service**: Retrieval-augmented generation
- **cultural_context**: Rwandan cultural context integration
- **llm_safety**: Safety and content filtering

## Service Dependencies

Services are initialized in dependency order:

1. **llm_config** (no dependencies)
2. **database** (no dependencies)
3. **session_manager** (depends on database)
4. **llm_service** (depends on llm_config)
5. **query_validator** (depends on llm_service)
6. **crisis_interceptor** (no dependencies)
7. **emotion_classifier** (depends on llm_service)
8. **retriever_service** (depends on database)
9. **rag_service** (depends on retriever_service, llm_service)
10. **state_router** (depends on session_manager, crisis_interceptor)
11. **langgraph_state_router** (depends on llm_service, session_manager, crisis_interceptor)
12. **cultural_context** (no dependencies)
13. **llm_safety** (depends on llm_config)

## Error Handling

The service container gracefully handles missing services:

- Optional services (marked as `required=False`) will log warnings but won't fail initialization
- Required services must initialize successfully or the entire initialization fails
- Import errors are caught and logged with fallback to None for missing services

## Health Monitoring

Check service health status:

```python
from backend.app.services import check_service_health

health_status = await check_service_health()
for service_name, status in health_status.items():
    print(f"{service_name}: {'✅' if status['healthy'] else '❌'}")
```

## Environment Variables

Configure services using environment variables:

```bash
# LLM Configuration
export LLM_TEMPERATURE=0.95
export LLM_MAX_TOKENS=512
export DEFAULT_MODEL_NAME=llama3.2:1b

# Database
export DATABASE_URL=postgresql://user:pass@localhost/mindora

# External APIs
export OPENAI_API_KEY=your_key
export GROQ_API_KEY=your_key
export OLLAMA_BASE_URL=http://localhost:11434
```

## Running the Initialization Script

Test service initialization:

```bash
# From the project root
python -m backend.app.services.initialize_services

# Or directly
python backend/app/services/initialize_services.py
```

## Custom Service Registration

Register additional services:

```python
from backend.app.services.service_container import ServiceConfig

# Define your service
def create_my_service():
    return MyCustomService()

# Register it
service_container.registry.register_service(
    "my_service",
    create_my_service,
    ServiceConfig(
        name="my_service",
        dependencies=["llm_service"],  # List dependencies
        required=True
    )
)
```

## Best Practices

1. **Always use the service lifecycle context manager** for proper cleanup
2. **Check service availability** before using them in production
3. **Handle optional services gracefully** in your application logic
4. **Use dependency injection** in FastAPI routes for better testability
5. **Monitor service health** in production environments
6. **Configure services** using environment variables for different environments

## Troubleshooting

### Common Issues

1. **Service initialization fails**
   - Check that all dependencies are available
   - Verify environment variables are set correctly
   - Check logs for detailed error messages

2. **Service not found**
   - Ensure the service is registered in `_register_default_services()`
   - Check for typos in service names
   - Verify the service class exists and can be imported

3. **Import errors**
   - Some services may not be implemented yet (marked as optional)
   - Check that required dependencies are installed
   - Verify import paths are correct

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration Examples

See `integration_example.py` for complete FastAPI integration examples and `initialize_services.py` for standalone usage patterns.