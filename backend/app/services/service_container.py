#  backend/app/services/service_container.py
"""
Global Service Container for centralized service initialization and management.

This module provides a centralized way to initialize, configure, and access all
services in the application with proper dependency injection and lifecycle management.
"""

import asyncio
import os
import logging
from typing import Dict, Any, Optional, TypeVar, Type, Generic, List, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import asyncio
import os
import logging
from typing import Dict, Any, Optional, TypeVar, Type, Generic, List
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from ..settings.settings import settings
from .llm_database_operations import DatabaseManager as LLMDatabaseOperations # Renamed to avoid conflict
from .session_state_manager import SessionStateManager, session_manager
from .llm_providers import LLMProviderFactory
from .emotion_classifier import LLMEmotionClassifier, initialize_emotion_classifier, classify_emotion_sync
from .unified_rag_service import UnifiedRAGService, create_unified_rag_service
from .stateful_pipeline import StatefulMentalHealthPipeline
from .llm_cultural_context import RwandaCulturalManager, ResponseApproachManager, ConversationContextManager
from .llm_safety import SafetyManager

# Try to import CrisisAlertService, fallback to None if not available
try:
    from .crisis_alert_service_temporary import CrisisAlertService
except ImportError:
    CrisisAlertService = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ServiceConfig:
    """Configuration for service initialization."""
    name: str
    dependencies: list[str] = field(default_factory=list)
    required: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


class ServiceRegistry:
    """Registry for service factories and configurations."""

    def __init__(self):
        self._factories: Dict[str, Callable] = {}
        self._configs: Dict[str, ServiceConfig] = {}
        self._instances: Dict[str, Any] = {}
        self._initializing: set[str] = set()
        self._initialized: set[str] = set()

    def register_service(
        self,
        name: str,
        factory: Callable,
        config: Optional[ServiceConfig] = None
    ):
        """Register a service factory with its configuration."""
        self._factories[name] = factory
        if config is None:
            config = ServiceConfig(name=name)
        self._configs[name] = config
        logger.info(f"Registered service: {name}")

    def get_service(self, name: str) -> Any:
        """Get a service instance by name."""
        if name not in self._instances:
            raise RuntimeError(f"Service '{name}' not initialized. Call initialize() first.")
        return self._instances[name]

    def is_initialized(self, name: str) -> bool:
        """Check if a service is initialized."""
        return name in self._initialized

    def get_initialized_services(self) -> Dict[str, Any]:
        """Get all initialized services."""
        return {name: self._instances[name] for name in self._initialized}


class ServiceContainer:
    """Main service container for managing all application services."""

    def __init__(self):
        self.registry = ServiceRegistry()
        self._lock = asyncio.Lock()
        self._register_default_services()

    def _register_default_services(self):
        """Register all default services with their dependencies."""

        # Core services (no dependencies)
        self.registry.register_service(
            "llm_config",
            lambda: settings, # Directly return settings
            ServiceConfig("llm_config", required=True)
        )

        # Database and storage services
        self.registry.register_service(
            "database",
            lambda: self._create_database_service(),
            ServiceConfig("database", required=True)
        )

        # Session management
        self.registry.register_service(
            "session_manager",
            lambda: self._create_session_manager(),
            ServiceConfig("session_manager", ["database"], required=True)
        )

        # LLM services (moved to after RAG service)


        # Crisis detection now handled by stateful pipeline

        # LLM Provider service (core dependency for other services)
        self.registry.register_service(
            "llm_provider",
            lambda: self._create_llm_provider(),
            ServiceConfig("llm_provider", [], required=True)
        )

        # Emotion classification (LLM-powered standalone API)
        self.registry.register_service(
            "emotion_classifier",
            lambda: self._create_emotion_classifier(),
            ServiceConfig("emotion_classifier", ["llm_provider"], required=False)
        )

        # Unified RAG service
        self.registry.register_service(
            "unified_rag_service",
            lambda: self._create_unified_rag_service(),
            ServiceConfig("unified_rag_service", [], required=False)
        )

        # Crisis alert service
        self.registry.register_service(
            "crisis_alert_service",
            lambda: self._create_crisis_alert_service(),
            ServiceConfig("crisis_alert_service", [], required=False)
        )



        # Stateful mental health pipeline
        self.registry.register_service(
            "stateful_pipeline",
            lambda: self._create_stateful_pipeline(),
            ServiceConfig("stateful_pipeline", ["llm_provider", "unified_rag_service"], required=True)
        )

        # Cultural context
        self.registry.register_service(
            "cultural_context",
            lambda: self._create_cultural_context(),
            ServiceConfig("cultural_context", required=True)
        )

        # Safety services
        self.registry.register_service(
            "llm_safety",
            lambda: self._create_llm_safety(),
            ServiceConfig("llm_safety", ["llm_config"], required=True)
        )

    def _create_llm_config(self):
        """Create LLM configuration service."""
        return settings

    def _create_database_service(self):
        """Create database service."""
        try:
            return LLMDatabaseOperations()
        except ImportError:
            logger.warning("LLMDatabaseOperations not available, using mock")
            return None

    def _create_session_manager(self):
        """Create session state manager."""
        return session_manager

    def _create_llm_provider(self):
        """Create LLM provider directly."""
        try:
            provider = LLMProviderFactory.create_provider(
                provider_name=os.getenv("PROVIDER"),
                model_name=os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM3-3B")
            )
            logger.info(f"✅ LLM provider '{provider.provider_name}' created successfully")
            return provider
        except Exception as e:
            logger.error(f"Failed to create LLM provider: {e}")
            raise


    # Legacy crisis interceptor creation method removed

    def _create_emotion_classifier(self):
        """Create LLM-powered emotion classifier service."""
        try:
            # Get LLM provider for emotion classification
            llm_provider = self.get_service("llm_provider")
            if not llm_provider:
                logger.warning("LLM provider not available for emotion classifier")
                return None
            
            # Create LLM emotion classifier
            emotion_classifier = LLMEmotionClassifier(llm_provider)
            logger.info("🧠 LLM-powered emotion classifier created successfully")
            return emotion_classifier
            
        except Exception as e:
            logger.warning(f"LLM emotion classifier creation failed: {e}")
            return None

    def _create_unified_rag_service(self):
        """Create unified RAG service."""
        try:
            rag_service = create_unified_rag_service()
            if rag_service:
                logger.info("🔍 Unified RAG service created successfully")
                return rag_service
            else:
                logger.warning("Failed to create unified RAG service")
                return None
        except Exception as e:
            logger.warning(f"Unified RAG service creation failed: {e}")
            return None

    def _create_crisis_alert_service(self):
        """Create crisis alert service."""
        if CrisisAlertService is None:
            logger.warning("CrisisAlertService not available, skipping creation")
            return None
        try:
            crisis_alert_service = CrisisAlertService()
            logger.info("🚨 Crisis alert service created successfully")
            return crisis_alert_service
        except Exception as e:
            logger.warning(f"Crisis alert service creation failed: {e}")
            return None

    def _create_stateful_pipeline(self):
        """Create stateful mental health pipeline service."""
        try:
            # Get LLM provider
            llm_provider = self.get_service("llm_provider")
            if not llm_provider:
                logger.warning("LLM provider not available for stateful_pipeline, using default")
                return StatefulMentalHealthPipeline()
            
            # Get RAG service if available
            rag_service = None
            try:
                rag_service = self.get_service("unified_rag_service")
                if rag_service:
                    logger.info("🔍 RAG service injected into stateful pipeline")
            except Exception:
                logger.info("RAG service not available, continuing without it")
            
            return StatefulMentalHealthPipeline(
                llm_provider=llm_provider,
                rag_service=rag_service
            )
        except ImportError as e:
            logger.warning(f"Stateful pipeline not available: {e}")
            # Return a mock service
            class MockStatefulPipeline:
                def __init__(self):
                    self.initialized = False

                async def process_query(self, query, user_id=None, conversation_history=None):
                    return {
                        "response": "I'm here to support you. How can I help you today?",
                        "response_confidence": 0.5,
                        "response_reason": "Mock response",
                        "processing_metadata": [],
                        "errors": []
                    }

                def health_check(self):
                    return self.initialized

            return MockStatefulPipeline()

    def _create_cultural_context(self):
        """Create cultural context service."""
        try:
            class CulturalContextService:
                """Wrapper class for cultural context functionality."""

                def __init__(self):
                    self.cultural_manager = RwandaCulturalManager()
                    self.response_manager = ResponseApproachManager()
                    self.context_manager = ConversationContextManager()

                def get_crisis_resources(self) -> Dict[str, Any]:
                    """Get Rwanda-specific crisis resources."""
                    return self.cultural_manager.get_crisis_resources()

                def get_cultural_context(self) -> Dict[str, str]:
                    """Get Rwanda-specific cultural context."""
                    return self.cultural_manager.get_cultural_context()

                def get_response_approach(self, emotion: str, user_message: str, conversation_context: List[Dict[str, str]]) -> Dict[str, str]:
                    """Get contextual response approach."""
                    return self.response_manager.get_contextual_response_approach(emotion, user_message, conversation_context)

                def build_system_prompt(self, context_parts: List[str], emotion: str, response_approach: Dict[str, str]) -> str:
                    """Build contextual system prompt."""
                    return self.response_manager.build_system_prompt(context_parts, emotion, response_approach)

                def health_check(self) -> bool:
                    """Check if cultural context service is healthy."""
                    return True

            return CulturalContextService()

        except ImportError:
            logger.warning("CulturalContextService not available, using mock")
            return None

    def _create_llm_safety(self):
        """Create LLM safety service."""
        try:
            class LLMSafety:
                """Wrapper class for LLM safety functionality."""

                def __init__(self):
                    self.safety_manager = SafetyManager()

                def initialize(self, chat_model=None):
                    """Initialize the safety service."""
                    try:
                        # Safety manager doesn't need chat model initialization
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to initialize LLM safety: {e}")
                        return False

                def sanitize_input(self, user_message: str) -> str:
                    """Sanitize user input."""
                    return self.safety_manager.sanitize_input(user_message)

                def is_safe_output(self, response: str) -> bool:
                    """Check if output is safe."""
                    return self.safety_manager.is_safe_output(response)

                def classify_intent(self, user_message: str) -> str:
                    """Classify user intent."""
                    return self.safety_manager.classify_intent(user_message)

                async def check_safety(self, message: str) -> Optional[str]:
                    """Check message safety and return appropriate response if needed."""
                    return self.safety_manager.check_safety(message)

                def health_check(self) -> bool:
                    """Check if LLM safety service is healthy."""
                    return True

            return LLMSafety()

        except ImportError:
            logger.warning("LLMSafety not available, using mock")
            return None

    async def initialize_service(self, name: str) -> bool:
        """Initialize a single service and its dependencies."""
        async with self._lock:
            return await self._initialize_service_recursive(name)

    async def _initialize_service_recursive(self, name: str) -> bool:
        """Recursively initialize a service and its dependencies."""
        if name in self.registry._initialized:
            return True

        if name in self.registry._initializing:
            logger.warning(f"Service '{name}' is already being initialized")
            return False

        if name not in self.registry._factories:
            logger.error(f"Service '{name}' not registered")
            return False

        config = self.registry._configs[name]

        # Initialize dependencies first
        for dep in config.dependencies:
            if not await self._initialize_service_recursive(dep):
                if config.required:
                    logger.error(f"Failed to initialize dependency '{dep}' for service '{name}'")
                    return False
                else:
                    logger.warning(f"Skipping optional dependency '{dep}' for service '{name}'")

        # Initialize the service
        try:
            self.registry._initializing.add(name)
            logger.info(f"Initializing service: {name}")

            factory = self.registry._factories[name]
            instance = factory()

            # If the instance has an initialize method, call it (sync or async)
            if hasattr(instance, 'initialize'):
                if asyncio.iscoroutinefunction(instance.initialize):
                    await instance.initialize()
                else:
                    # Call synchronous initialize method
                    instance.initialize()

            self.registry._instances[name] = instance
            self.registry._initialized.add(name)

            logger.info(f"Successfully initialized service: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize service '{name}': {e}")
            return False
        finally:
            self.registry._initializing.discard(name)

    async def initialize_all_services(self) -> bool:
        """Initialize all registered services."""
        logger.info("Starting global service initialization...")

        success = True
        for name in self.registry._factories.keys():
            if not await self.initialize_service(name):
                if self.registry._configs[name].required:
                    success = False
                    logger.error(f"Failed to initialize required service: {name}")
                else:
                    logger.warning(f"Failed to initialize optional service: {name}")

        if success:
            logger.info("✅ All services initialized successfully")
        else:
            logger.error("❌ Some services failed to initialize")

        return success

    def get_service(self, name: str) -> Any:
        """Get a service instance by name."""
        return self.registry.get_service(name)

    def is_initialized(self, name: str) -> bool:
        """Check if a service is initialized."""
        return self.registry.is_initialized(name)

    def get_initialized_services(self) -> Dict[str, Any]:
        """Get all initialized services."""
        return self.registry.get_initialized_services()

    async def shutdown_all_services(self):
        """Shutdown all initialized services."""
        logger.info("Shutting down all services...")

        for name in list(self.registry._initialized):
            try:
                instance = self.registry._instances[name]

                # If the instance has shutdown or close methods, call them (sync or async)
                if hasattr(instance, 'shutdown'):
                    if asyncio.iscoroutinefunction(instance.shutdown):
                        await instance.shutdown()
                    else:
                        instance.shutdown()
                elif hasattr(instance, 'close'):
                    if asyncio.iscoroutinefunction(instance.close):
                        await instance.close()
                    else:
                        instance.close()

                logger.info(f"Shutdown service: {name}")
            except Exception as e:
                logger.error(f"Error shutting down service '{name}': {e}")

        self.registry._instances.clear()
        self.registry._initialized.clear()
        logger.info("All services shutdown complete")


# Global service container instance
service_container = ServiceContainer()


# Convenience functions for easy access
def get_service(name: str) -> Any:
    """Get a service instance by name."""
    return service_container.get_service(name)


def initialize_services() -> asyncio.Task:
    """Initialize all services asynchronously."""
    return asyncio.create_task(service_container.initialize_all_services())


async def shutdown_services():
    """Shutdown all services."""
    await service_container.shutdown_all_services()


@asynccontextmanager
async def service_lifecycle():
    """Async context manager for service lifecycle."""
    try:
        success = await service_container.initialize_all_services()
        if not success:
            raise RuntimeError("Failed to initialize all services")
        yield service_container
    finally:
        await service_container.shutdown_all_services()


# Service health check
async def check_service_health() -> Dict[str, Any]:
    """Check the health status of all services."""
    health_status = {}

    for name in service_container.registry._factories.keys():
        try:
            instance = service_container.get_service(name)
            is_healthy = True

            # Check if service has a health check method
            if hasattr(instance, 'health_check'):
                try:
                    if asyncio.iscoroutinefunction(instance.health_check):
                        is_healthy = await instance.health_check()
                    else:
                        is_healthy = instance.health_check()
                except Exception as e:
                    logger.warning(f"Health check failed for {name}: {e}")
                    is_healthy = False
            else:
                # If no health check method, assume healthy if initialized
                is_healthy = True

            health_status[name] = {
                "initialized": service_container.is_initialized(name),
                "healthy": is_healthy
            }

        except Exception as e:
            health_status[name] = {
                "initialized": False,
                "healthy": False,
                "error": str(e)
            }

    return health_status