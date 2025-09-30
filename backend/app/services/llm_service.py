"""
Refactored LLM Service - Main orchestrator for mental health chatbot.

This module provides a clean, maintainable interface for LLM operations
while delegating specific responsibilities to focused modules.
"""
import time
from typing import Dict, List, Any, Optional
from langchain.schema import HumanMessage, SystemMessage

from .llm_config import (
    MAX_INPUT_LENGTH, RAG_TOP_K, FALLBACK_RESPONSE,
    ERROR_MESSAGES
)
from .llm_safety import SafetyManager, GuardrailsManager
from .llm_cultural_context import (
    RwandaCulturalManager,
    ResponseApproachManager,
    ConversationContextManager
)
from .llm_providers import LLMProviderFactory, create_llm_provider
from .llm_database_operations import DatabaseManager
from .retriever_service import RetrieverService
from .emotion_classifier import classify_emotion
from .chatbot_insights_pipeline import detect_medication_mentions, detect_suicide_risk


class LLMService:
    """
    Main LLM service orchestrator for mental health conversations.

    This class provides a clean interface while delegating specific
    responsibilities to focused, maintainable modules.
    """

    def __init__(self, model_name: Optional[str] = None, provider_name: Optional[str] = None, use_vllm: bool = False):
        """Initialize the LLM service components (lightweight constructor)."""
        # Store configuration
        self.model_name = model_name or "HuggingFaceTB/SmolLM3-3B"
        self.provider_name = provider_name
        self.use_vllm = use_vllm

        # Initialize components (but don't start them yet)
        self.llm_provider =  None
        self.retriever = RetrieverService()
        self.guardrails_manager = None

        # Status flags
        self._is_initialized = False
        self._initialization_error = None

    async def generate_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        skip_analysis: bool = False,
        emotion_data: Optional[Dict[str, Any]] = None,
        user_gender: Optional[str] = None
    ) -> str:
        """
        Generate a response to a user message with full pipeline processing.

        Args:
            user_message: The user's input message
            conversation_history: Previous conversation messages
            user_id: User identifier for conversation context
            skip_analysis: Skip expensive analysis for simple messages
            emotion_data: Emotion analysis data from LangGraph
            user_gender: User's gender for personalized responses

        Returns:
            Generated response string
        """
        pipeline_start = time.time()
        print(f"\n🚀 Starting message pipeline for user {user_id}")

        # Check if service is initialized
        if not self._is_initialized:
            if self._initialization_error:
                print(f"⚠️  LLM service not initialized: {self._initialization_error}")
                print(f"   Using contextual fallback responses with query validation context")
                # Don't return error - allow fallback responses
            else:
                print(f"⚠️  LLM service not initialized, using fallback responses")
                # Don't return error - allow fallback responses

        # Sanitize input
        sanitized_message = SafetyManager.sanitize_input(user_message)
        print(f"    🧹 LLM: Input sanitized ({len(user_message)} -> {len(sanitized_message)} chars)")

        # Apply guardrails first
        if self.guardrails_manager:
            guardrails_start = time.time()
            guardrails_response = await self.guardrails_manager.check_guardrails(sanitized_message)
            guardrails_time = time.time() - guardrails_start
            print(f"    🛡️  LLM: Guardrails check: {guardrails_time:.3f}s")

            if guardrails_response:
                return guardrails_response

        # Use sanitized message for processing
        user_message = sanitized_message

        # Determine if we should skip expensive analysis
        if ConversationContextManager.should_skip_analysis(user_message, skip_analysis):
            return await self._handle_fast_path(user_message, pipeline_start, emotion_data, user_gender)

        # Run full analysis pipeline
        return await self._handle_full_analysis(
            user_message, conversation_history, user_id, pipeline_start, emotion_data, user_gender
        )

    async def _handle_fast_path(self, user_message: str, pipeline_start: float, emotion_data: Optional[Dict[str, Any]] = None, user_gender: Optional[str] = None) -> str:
        """Handle simple messages with minimal processing."""
        print("    📝 LLM: Using fast path (short message)")

        # Use emotion data from LangGraph if provided, otherwise use neutral
        if emotion_data:
            emotion = emotion_data.get("detected_emotion", "neutral")
            print(f"    🎭 LLM: Using emotion data from LangGraph: {emotion}")
        else:
            emotion = "neutral"

        # Build basic context
        context_parts = ["This appears to be a new conversation."]
        response_approach = ResponseApproachManager.get_contextual_response_approach(
            emotion, user_message, []
        )

        # Build system prompt
        system_prompt = ResponseApproachManager.build_system_prompt(
            context_parts, emotion, response_approach
        )

        # Generate response
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        if not self.llm_provider:
            raise RuntimeError("LLM provider not initialized")
        response = await self.llm_provider.generate_response(messages)

        # Apply safety filtering
        if not SafetyManager.is_safe_output(response):
            response = FALLBACK_RESPONSE

        total_time = time.time() - pipeline_start
        print(f"🏁 Fast path total time: {total_time:.3f}s")

        return response

    async def _handle_full_analysis(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        user_id: Optional[str],
        pipeline_start: float,
        emotion_data: Optional[Dict[str, Any]] = None,
        user_gender: Optional[str] = None
    ) -> str:
        """Handle complex messages with full analysis pipeline."""
        # Analysis pipeline
        analysis_start = time.time()

        # Use emotion data from LangGraph if provided, otherwise detect locally
        if emotion_data:
            emotion = emotion_data.get("detected_emotion", "neutral")
            print(f"    🎭 LLM: Using emotion data from LangGraph: {emotion}")
        else:
            # Try to classify emotion locally, fallback to neutral if not available
            try:
                emotion = classify_emotion(user_message)
                print(f"    🎭 LLM: Local emotion classification: {emotion}")
            except RuntimeError:
                # Emotion classifier not initialized, use neutral as fallback
                emotion = "neutral"
                print(f"    🎭 LLM: Emotion classifier not available, using neutral")

        suicide_flag = detect_suicide_risk(user_message)
        meds_mentioned = detect_medication_mentions(user_message)
        analysis_time = time.time() - analysis_start
        print(f"    📊 LLM: Analysis pipeline: {analysis_time:.3f}s")

        # RAG retrieval (skip for simple greetings)
        retrieved_text = ""
        if not ConversationContextManager.is_simple_greeting(user_message):
            rag_start = time.time()
            try:
                retrieved_chunks = self.retriever.search(query=user_message, top_k=RAG_TOP_K)
                retrieved_text = "\n\n".join(
                    chunk for chunk in retrieved_chunks if isinstance(chunk, str)
                ) if retrieved_chunks else ""
                rag_time = time.time() - rag_start
                print(f"    🔍 LLM: RAG search: {rag_time:.3f}s ({len(retrieved_chunks)} chunks)")
            except Exception as e:
                print(f"    ❌ LLM: RAG Error: {e}")
                retrieved_text = ""

        # Get conversation history if not provided
        if not conversation_history and user_id:
            conversation_history = DatabaseManager.fetch_recent_conversation(user_id)

        # Build context and response approach
        prompt_start = time.time()
        memory_block = ConversationContextManager.build_memory_block(conversation_history or [])

        # Get contextual response approach
        response_approach = ResponseApproachManager.get_contextual_response_approach(
            emotion, user_message, conversation_history or []
        )

        # Build context parts
        context_parts = []
        if memory_block.strip():
            context_parts.append(f"Previous conversation context:\n{memory_block}")
        else:
            context_parts.append("This appears to be a new conversation.")

        if suicide_flag:
            context_parts.append("⚠️ CRISIS INDICATOR: Suicide risk detected - prioritize safety and professional referral")

        if meds_mentioned:
            context_parts.append(f"📋 Medications mentioned: {', '.join(meds_mentioned)} - be mindful of medication safety")

        if retrieved_text:
            context_parts.append(f"Relevant knowledge: {retrieved_text[:500]}")

        # Build system prompt
        system_prompt = ResponseApproachManager.build_system_prompt(
            context_parts, emotion, response_approach
        )

        # Generate response
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        prompt_time = time.time() - prompt_start
        print(f"    📝 LLM: Prompt building: {prompt_time:.3f}s ({len(system_prompt)} chars)")

        model_start = time.time()
        if not self.llm_provider:
            raise RuntimeError("LLM provider not initialized")
        response = await self.llm_provider.generate_response(messages)
        model_time = time.time() - model_start
        print(f"    🤖 LLM: Model inference: {model_time:.3f}s")

        # Apply safety filtering
        if not SafetyManager.is_safe_output(response):
            print("    ⚠️  LLM: Unsafe output detected, using fallback response")
            response = FALLBACK_RESPONSE

        total_time = time.time() - pipeline_start
        print(f"🏁 Full analysis total time: {total_time:.3f}s")

        return response

    def get_rwanda_crisis_resources(self) -> Dict[str, Any]:
        """Get Rwanda-specific crisis resources."""
        return RwandaCulturalManager.get_crisis_resources()

    def get_rwanda_cultural_context(self) -> Dict[str, str]:
        """Get Rwanda-specific cultural context."""
        return RwandaCulturalManager.get_cultural_context()

    def get_grounding_exercise(self) -> str:
        """Get culturally resonant grounding exercise."""
        return RwandaCulturalManager.get_grounding_exercise()

    def initialize(self) -> bool:
        """
        Initialize the LLM service components.
        This should be called once during application startup.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            print("🚀 Initializing LLM Service...")

            # Determine provider based on configuration
            if self.use_vllm:
                # For vLLM, we still use Ollama provider but with different base URL
                provider_name = "ollama"
                base_url = "http://127.0.0.1:8000"  # vLLM default
            else:
                # Use the specified provider or auto-detect
                provider_name = self.provider_name or "ollama"

            # Create LLM provider using factory
            try:
                self.llm_provider = LLMProviderFactory.create_provider(
                    provider_name=provider_name,
                    model_name=self.model_name
                )
                print(f"✅ LLM provider '{provider_name}' initialized successfully")
            except Exception as e:
                self._initialization_error = f"Failed to create LLM provider: {e}"
                print(f"❌ {self._initialization_error}")
                return False

            # Initialize guardrails if provider is ready
            if self.llm_provider.is_available():
                try:
                    # Get the underlying chat model from the provider for guardrails
                    # We need to access the internal chat_model attribute
                    chat_model = getattr(self.llm_provider, '_chat_model', None)
                    if chat_model:
                        self.guardrails_manager = GuardrailsManager(chat_model)
                        print("✅ Guardrails initialized successfully")
                    else:
                        print("⚠️  Guardrails skipped - no chat model available yet")
                except Exception as e:
                    self._initialization_error = f"Failed to initialize guardrails: {e}"
                    print(f"⚠️  {self._initialization_error}")
                    # Continue without guardrails if they fail
            else:
                self._initialization_error = "Provider not available - will use fallback responses"
                print(f"⚠️  {self._initialization_error}")
                print(f"   Model '{self.model_name}' not found or server not running")

            # Test retriever connection
            try:
                # Simple test to ensure retriever can connect
                test_query = "test"
                self.retriever.search(test_query, top_k=1)
                print("✅ Vector retriever initialized successfully")
            except Exception as e:
                self._initialization_error = f"Failed to initialize vector retriever: {e}"
                print(f"⚠️  {self._initialization_error}")
                # Continue without RAG if it fails

            self._is_initialized = True
            print("✅ LLM Service initialization completed")
            return True

        except Exception as e:
            self._initialization_error = f"LLM Service initialization failed: {e}"
            print(f"❌ {self._initialization_error}")
            self._is_initialized = False
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if the service is properly initialized."""
        return getattr(self, '_is_initialized', False)

    @property
    def initialization_error(self) -> Optional[str]:
        """Get the last initialization error message."""
        return getattr(self, '_initialization_error', None)