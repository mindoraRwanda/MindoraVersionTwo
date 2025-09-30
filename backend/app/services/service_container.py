"""
Global Service Container for centralized service initialization and management.

This module provides a centralized way to initialize, configure, and access all
services in the application with proper dependency injection and lifecycle management.
"""

import asyncio
import os
import logging
from typing import Dict, Any, Optional, TypeVar, Type, Generic, List
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

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
        self._factories: Dict[str, callable] = {}
        self._configs: Dict[str, ServiceConfig] = {}
        self._instances: Dict[str, Any] = {}
        self._initializing: set[str] = set()
        self._initialized: set[str] = set()

    def register_service(
        self,
        name: str,
        factory: callable,
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
            lambda: self._create_llm_config(),
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

        # LLM services
        self.registry.register_service(
            "llm_service",
            lambda: self._create_llm_service(),
            ServiceConfig("llm_service", ["llm_config"], required=True)
        )

        # Query validation and processing
        self.registry.register_service(
            "query_validator",
            lambda: self._create_query_validator(),
            ServiceConfig("query_validator", ["llm_service"], required=True)
        )

        # Crisis detection
        self.registry.register_service(
            "crisis_interceptor",
            lambda: self._create_crisis_interceptor(),
            ServiceConfig("crisis_interceptor", required=True)
        )

        # Emotion classification
        self.registry.register_service(
            "emotion_classifier",
            lambda: self._create_emotion_classifier(),
            ServiceConfig("emotion_classifier", ["llm_service"], required=False)
        )

        # RAG services
        self.registry.register_service(
            "retriever_service",
            lambda: self._create_retriever_service(),
            ServiceConfig("retriever_service", ["database"], required=False)
        )

        self.registry.register_service(
            "rag_service",
            lambda: self._create_rag_service(),
            ServiceConfig("rag_service", ["retriever_service", "llm_service"], required=False)
        )

        # State management
        self.registry.register_service(
            "state_router",
            lambda: self._create_state_router(),
            ServiceConfig("state_router", ["session_manager", "crisis_interceptor"], required=True)
        )

        self.registry.register_service(
            "langgraph_state_router",
            lambda: self._create_langgraph_state_router(),
            ServiceConfig("langgraph_state_router", ["llm_service", "session_manager", "crisis_interceptor"], required=True)
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
        from .llm_config import config_manager
        return config_manager

    def _create_database_service(self):
        """Create database service."""
        try:
            from .llm_database_operations import LLMDatabaseOperations
            return LLMDatabaseOperations()
        except ImportError:
            logger.warning("LLMDatabaseOperations not available, using mock")
            return None

    def _create_session_manager(self):
        """Create session state manager."""
        from .session_state_manager import SessionStateManager, session_manager
        return session_manager

    def _create_llm_service(self):
        """Create LLM service."""
        from .llm_service import LLMService
        return LLMService(use_vllm=False, provider_name=os.getenv("PROVIDER_NAME"), model_name=os.getenv("PROVIDER_MODEL"))

    def _create_query_validator(self):
        """Create query validator service."""
        try:
            from .query_validator_langgraph import LangGraphQueryValidator
            # Check if LLM service is available for LangGraph validator
            if "llm_service" in self.registry._instances:
                llm_service = self.registry._instances["llm_service"]
                if llm_service and llm_service.llm_provider:
                    return LangGraphQueryValidator(llm_provider=llm_service.llm_provider)
                else:
                    # Fallback to basic validator if no LLM provider
                    from .query_validator import QueryValidatorService
                    return QueryValidatorService()
            else:
                # Fallback to basic validator if LLM service not available
                from .query_validator import QueryValidatorService
                return QueryValidatorService()
        except Exception as e:
            logger.warning(f"QueryValidator creation failed: {e}, using None")
            return None

    def _create_crisis_interceptor(self):
        """Create crisis interceptor service."""
        try:
            from .crisis_interceptor import CrisisInterceptor
            return CrisisInterceptor()
        except ImportError:
            logger.warning("CrisisInterceptor not available, using mock")
            return None

    def _create_emotion_classifier(self):
        """Create emotion classifier service."""
        try:
            from .emotion_classifier import classify_emotion, initialize_emotion_classifier

            class EmotionClassifier:
                """Wrapper class for emotion classification functionality."""

                def __init__(self):
                    self.model = None
                    self.emotion_embeddings = None

                def initialize(self):
                    """Initialize the emotion classifier."""
                    try:
                        self.model = initialize_emotion_classifier()
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to initialize emotion classifier: {e}")
                        return False

                def classify_emotion(self, text: str) -> str:
                    """Classify emotion in text."""
                    try:
                        return classify_emotion(text)
                    except Exception as e:
                        logger.warning(f"Emotion classification failed: {e}")
                        return "neutral"

                def health_check(self) -> bool:
                    """Check if emotion classifier is healthy."""
                    return self.model is not None

            classifier = EmotionClassifier()
            # Try to initialize but don't fail if it doesn't work
            try:
                classifier.initialize()
            except Exception:
                pass  # Continue with uninitialized classifier

            return classifier

        except ImportError:
            logger.warning("EmotionClassifier not available, using mock")
            return None

    def _create_retriever_service(self):
        """Create retriever service."""
        try:
            from .retriever_service import RetrieverService
            return RetrieverService()
        except ImportError:
            logger.warning("RetrieverService not available, using mock")
            return None

    def _create_rag_service(self):
        """Create RAG service."""
        try:
            from .rag_service import RAGService
            return RAGService()
        except ImportError:
            logger.warning("RAGService not available, using mock")
            return None

    def _create_state_router(self):
        """Create state router service."""
        try:
            from .state_router import StateRouter
            return StateRouter()
        except ImportError:
            logger.warning("StateRouter not available, using mock")
            return None

    def _create_langgraph_state_router(self):
        """Create LangGraph state router service."""
        from .langgraph_state_router import LLMEnhancedStateRouter

        # Check if LLM service is available in the registry
        if "llm_service" in self.registry._instances:
            llm_service = self.registry._instances["llm_service"]
            return LLMEnhancedStateRouter(llm_service=llm_service)
        else:
            # Fallback to default if LLM service not available yet
            logger.warning("LLM service not available for langgraph_state_router, using default")
            return LLMEnhancedStateRouter()

    def _create_cultural_context(self):
        """Create cultural context service."""
        try:
            from .llm_cultural_context import RwandaCulturalManager, ResponseApproachManager, ConversationContextManager

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
            from .llm_safety import SafetyManager, GuardrailsManager

            class LLMSafety:
                """Wrapper class for LLM safety functionality."""

                def __init__(self):
                    self.safety_manager = SafetyManager()
                    self.guardrails_manager = None

                def initialize(self, chat_model=None):
                    """Initialize the safety service."""
                    try:
                        if chat_model:
                            self.guardrails_manager = GuardrailsManager(chat_model)
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

                async def check_guardrails(self, message: str) -> Optional[str]:
                    """Check message against guardrails."""
                    if self.guardrails_manager:
                        return await self.guardrails_manager.check_guardrails(message)
                    return None

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