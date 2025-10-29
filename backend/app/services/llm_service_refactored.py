# services/llm_service_refactored.py
"""
Refactored LLM Service - Main orchestrator for mental health chatbot.

This module provides a clean, maintainable interface for LLM operations
while delegating specific responsibilities to focused modules.

NOTE: Crisis detection, logging, and therapist notification are handled
by the API router (LLM crisis classifier + safety_pipeline), not here.
"""

import time
from typing import Dict, List, Any, Optional

from langchain.schema import HumanMessage, SystemMessage

from .llm_config import (
    MAX_INPUT_LENGTH, RAG_TOP_K, FALLBACK_RESPONSE,
    ERROR_MESSAGES
)
from .llm_cultural_context import (
    RwandaCulturalManager,
    ResponseApproachManager,
    ConversationContextManager
)
from .llm_providers import LLMProviderFactory
from .llm_database_operations import DatabaseManager
from .retriever_service import RetrieverService
from .emotion_classifier import classify_emotion
from .chatbot_insights_pipeline import detect_medication_mentions, detect_suicide_risk


class LLMService:
    """
    Main LLM service orchestrator for mental health conversations.

    Responsibilities:
      • Prompt building with cultural/context blocks
      • Optional RAG retrieval
      • Delegating generation to the configured provider

    Crisis decisions + logging/email are handled outside (router + safety_pipeline).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider_name: Optional[str] = None,
        use_vllm: bool = False
    ):
        # Config
        self.model_name = model_name or "gemma3:1b"
        self.provider_name = provider_name
        self.use_vllm = use_vllm

        # Components
        self.llm_provider = None
        self.retriever = RetrieverService()

        # Status
        self._is_initialized: bool = False
        self._initialization_error: Optional[str] = None

    async def generate_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        skip_analysis: bool = False,
        emotion_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response to a user message with full pipeline processing.
        """
        pipeline_start = time.time()
        print(f"\n🚀 Starting message pipeline for user {user_id}")

        # Initialization guard
        if not self._is_initialized:
            if self._initialization_error:
                return f"Service not initialized: {self._initialization_error}"
            return ERROR_MESSAGES["model_not_initialized"]

        # Minimal input cleaning (no SafetyManager)
        sanitized_message = (user_message or "").strip()
        if isinstance(MAX_INPUT_LENGTH, int) and MAX_INPUT_LENGTH > 0:
            sanitized_message = sanitized_message[:MAX_INPUT_LENGTH]
        print(f"    🧹 LLM: Input prepped ({len(user_message or '')} -> {len(sanitized_message)} chars)")

        # Decide path
        if ConversationContextManager.should_skip_analysis(sanitized_message, skip_analysis):
            return await self._handle_fast_path(sanitized_message, pipeline_start, emotion_data)

        # Full analysis
        return await self._handle_full_analysis(
            sanitized_message, conversation_history, user_id, pipeline_start, emotion_data
        )

    async def _handle_fast_path(
        self,
        user_message: str,
        pipeline_start: float,
        emotion_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle simple messages with minimal processing."""
        print("    📝 LLM: Using fast path (short message)")

        # Emotion from upstream (LangGraph) or default
        emotion = (emotion_data or {}).get("detected_emotion", "neutral")

        # Basic context
        context_parts = ["This appears to be a new conversation."]
        response_approach = ResponseApproachManager.get_contextual_response_approach(
            emotion, user_message, []
        )

        # System prompt
        system_prompt = ResponseApproachManager.build_system_prompt(
            context_parts, emotion, response_approach
        )

        # Generate
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        response = await self.llm_provider.generate_response(messages)

        # Minimal post-check
        if not response or not str(response).strip():
            response = FALLBACK_RESPONSE

        print(f"🏁 Fast path total time: {time.time() - pipeline_start:.3f}s")
        return response

    async def _handle_full_analysis(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        user_id: Optional[str],
        pipeline_start: float,
        emotion_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle complex messages with full analysis pipeline."""
        analysis_start = time.time()

        # Emotion: prefer upstream if provided
        emotion = (emotion_data or {}).get("detected_emotion") or classify_emotion(user_message)

        # Lightweight insight flags (not for crisis decisions; router handles crisis)
        suicide_flag = detect_suicide_risk(user_message)
        meds_mentioned = detect_medication_mentions(user_message)
        print(f"    📊 LLM: Analysis pipeline: {time.time() - analysis_start:.3f}s")

        # RAG retrieval (skip simple greetings)
        retrieved_text = ""
        if not ConversationContextManager.is_simple_greeting(user_message):
            try:
                rag_start = time.time()
                retrieved_chunks = self.retriever.search(query=user_message, top_k=RAG_TOP_K)
                retrieved_text = "\n\n".join(
                    chunk for chunk in (retrieved_chunks or []) if isinstance(chunk, str)
                )
                print(f"    🔍 LLM: RAG search: {time.time() - rag_start:.3f}s ({len(retrieved_chunks or [])} chunks)")
            except Exception as e:
                print(f"    ❌ LLM: RAG Error: {e}")

        # Fetch conversation history if not provided
        if not conversation_history and user_id:
            conversation_history = DatabaseManager.fetch_recent_conversation(user_id)

        # Prompt building
        prompt_start = time.time()
        memory_block = ConversationContextManager.build_memory_block(conversation_history or [])

        context_parts: List[str] = []
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

        response_approach = ResponseApproachManager.get_contextual_response_approach(
            emotion, user_message, conversation_history or []
        )
        system_prompt = ResponseApproachManager.build_system_prompt(
            context_parts, emotion, response_approach
        )
        print(f"    📝 LLM: Prompt building: {time.time() - prompt_start:.3f}s ({len(system_prompt)} chars)")

        # Generate
        model_start = time.time()
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        response = await self.llm_provider.generate_response(messages)
        print(f"    🤖 LLM: Model inference: {time.time() - model_start:.3f}s")

        # Minimal post-check
        if not response or not str(response).strip():
            response = FALLBACK_RESPONSE

        print(f"🏁 Full analysis total time: {time.time() - pipeline_start:.3f}s")
        return response

    def get_rwanda_crisis_resources(self) -> Dict[str, Any]:
        """Get Rwanda-specific crisis resources."""
        return RwandaCulturalManager.get_crisis_resources()

    def get_rwanda_cultural_context(self) -> Dict[str, str]:
        """Get Rwanda-specific cultural context."""
        return RwandaCulturalManager.get_cultural_context()

    def get_grounding_exercise(self) -> str:
        """Get Rwanda-culturally appropriate grounding exercise."""
        return RwandaCulturalManager.get_grounding_exercise()

    def initialize(self) -> bool:
        """
        Initialize the LLM service components.
        Call once during application startup.
        """
        try:
            print("🚀 Initializing LLM Service...")

            # Provider selection
            provider_name = self.provider_name or ("ollama" if not self.use_vllm else "ollama")

            # Create provider
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

            # Test retriever (best-effort)
            try:
                self.retriever.search("health-check", top_k=1)
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
        return bool(self._is_initialized)

    @property
    def initialization_error(self) -> Optional[str]:
        return self._initialization_error
