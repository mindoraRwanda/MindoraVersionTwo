"""
Stateful Mental Health Pipeline Orchestrator

This module implements the main LangGraph orchestrator for the unified mental health
chatbot pipeline with comprehensive state management and explainability.
"""

from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.orm import Session
from fastapi import BackgroundTasks
import asyncio
import time
import logging

from ..db.database import get_db

from .pipeline_state import (
    StatefulPipelineState, ResponseStrategy, CrisisSeverity, QueryType,
    create_initial_pipeline_state, add_processing_metadata, increment_llm_calls, add_error
)
from .pipeline_nodes import (
    QueryValidationNode, CrisisDetectionNode, EmotionDetectionNode,
    QueryEvaluationNode, ElaborationNode, EmpathyNode, ClarificationNode,
    SuggestionNode, GuidanceNode, IdleNode, CrisisAlertNode, GenerateResponseNode,
    UnifiedAnalysisNode,
)
from .rag_enhancement_node import RAGEnhancementNode
from ..settings import settings
from ..prompts.cultural_context_prompts import CulturalContextPrompts

logger = logging.getLogger(__name__)


class StatefulMentalHealthPipeline:
    """
    Main orchestrator for the stateful LangGraph mental health pipeline.
    
    This class manages the complete workflow from query validation through
    response generation with full explainability and state management.
    """
    
    def __init__(self, llm_provider=None, rag_service=None, db: Optional[Session] = None, background: Optional[BackgroundTasks] = None):
        """Initialize the stateful pipeline with LLM provider, RAG service, and safety parameters."""
        logger.info("🔧 Initializing StatefulMentalHealthPipeline...")
        self.llm_provider = llm_provider
        self.rag_service = rag_service
        self.db = db
        self.background = background
        self.cultural_prompts = CulturalContextPrompts()
        logger.info("📚 Cultural context prompts loaded")

        # Initialize node instances
        logger.info("Initializing pipeline nodes...")
        # Unified analysis replaces QueryValidation + CrisisDetection + QueryEvaluation
        self.unified_analysis_node = UnifiedAnalysisNode(self.llm_provider)
        self.emotion_detection_node = EmotionDetectionNode(self.llm_provider)  # local model, no LLM call
        self.empathy_node = EmpathyNode(self.llm_provider)
        self.crisis_alert_node = CrisisAlertNode(self.llm_provider)
        self.generate_response_node = GenerateResponseNode(self.llm_provider)
        # Legacy nodes kept for fallback but not wired into the main graph
        self.query_validation_node = QueryValidationNode(self.llm_provider)
        self.crisis_detection_node = CrisisDetectionNode(self.llm_provider)
        self.query_evaluation_node = QueryEvaluationNode(self.llm_provider)
        self.elaboration_node = ElaborationNode(self.llm_provider)
        self.clarification_node = ClarificationNode(self.llm_provider)
        self.suggestion_node = SuggestionNode(self.llm_provider)
        self.guidance_node = GuidanceNode(self.llm_provider)
        self.idle_node = IdleNode(self.llm_provider)
        
        # Initialize RAG enhancement node if RAG service is available
        if self.rag_service:
            self.rag_enhancement_node = RAGEnhancementNode(self.rag_service)
            logger.info("🔍 RAG enhancement node initialized")
        else:
            self.rag_enhancement_node = None
            logger.info("⚠️ RAG service not available, skipping RAG enhancement node")
        
        logger.info("✅ All pipeline nodes initialized")
        
        logger.info("🕸️  Building LangGraph workflow...")
        self.graph = self._build_pipeline_graph()
        logger.info("✅ StatefulMentalHealthPipeline initialization complete")
        
    def _build_pipeline_graph(self) -> StateGraph:
        """
        Simplified pipeline graph — 2 LLM calls per message (down from 5-6).

        Flow:
          emotion_detection (local model, no LLM)
            → unified_analysis  (1 LLM call: query type + crisis check)
            → [crisis_alert | rag_enhancement → empathy | generate_response]
            → generate_response → END
        """
        workflow = StateGraph(StatefulPipelineState)

        # ── Nodes ──────────────────────────────────────────────────────────────
        workflow.add_node("emotion_detection", self._emotion_detection_node)
        workflow.add_node("unified_analysis",  self._unified_analysis_node)
        workflow.add_node("empathy",           self._empathy_node)
        workflow.add_node("crisis_alert",      self._crisis_alert_node)
        workflow.add_node("generate_response", self._generate_response_node)

        if self.rag_enhancement_node:
            workflow.add_node("rag_enhancement", self._rag_enhancement_node)

        # ── Entry & fixed edges ────────────────────────────────────────────────
        workflow.set_entry_point("emotion_detection")
        workflow.add_edge("emotion_detection", "unified_analysis")

        if self.rag_enhancement_node:
            workflow.add_edge("rag_enhancement", "empathy")

        workflow.add_edge("empathy",        "generate_response")
        workflow.add_edge("crisis_alert",   "generate_response")
        workflow.add_edge("generate_response", END)

        # ── Conditional routing after unified_analysis ─────────────────────────
        routes = {
            "empathy":          "empathy",
            "crisis_alert":     "crisis_alert",
            "generate_response":"generate_response",
        }
        if self.rag_enhancement_node:
            routes["rag_enhancement"] = "rag_enhancement"

        workflow.add_conditional_edges(
            "unified_analysis",
            self._route_after_unified_analysis,
            routes,
        )

        logger.info("Pipeline graph built: emotion_detection → unified_analysis → therapy/crisis")
        return workflow.compile()
    
    def _route_after_unified_analysis(self, state: StatefulPipelineState) -> str:
        """Single routing decision after the unified analysis node."""
        validation = state.get("query_validation")
        crisis    = state.get("crisis_assessment")

        # 1. Non-mental-health → quick conversational reply
        non_therapy = {QueryType.GREETING, QueryType.CASUAL, QueryType.RANDOM}
        if validation and (validation.is_random or validation.query_type in non_therapy):
            logger.info(f"Routing to generate_response (type={validation.query_type.value})")
            return "generate_response"

        # 2. Genuine crisis (explicit suicidal ideation / active self-harm / imminent danger)
        if crisis and crisis.is_crisis:
            HIGH = {CrisisSeverity.SEVERE, CrisisSeverity.HIGH}
            if crisis.crisis_severity in HIGH and crisis.crisis_confidence >= 0.75:
                logger.warning(
                    f"Crisis: {crisis.crisis_severity.value} "
                    f"conf={crisis.crisis_confidence:.2f} → crisis_alert"
                )
                return "crisis_alert"
            logger.info(f"Low/medium crisis signal ({crisis.crisis_severity.value}) → therapy")

        # 3. Everything else → therapy session (with RAG if available)
        if self.rag_enhancement_node:
            return "rag_enhancement"
        return "empathy"
    
    async def process_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_gender: Optional[str] = None,
        db: Optional[Session] = None,
        background: Optional[BackgroundTasks] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete stateful pipeline.
        
        Args:
            query: User's input query
            user_id: Optional user identifier
            conversation_id: Optional conversation identifier
            message_id: Optional message identifier
            conversation_history: Optional conversation history
            user_gender: Optional user gender for cultural context (male, female, other, prefer_not_to_say)
            db: Optional database session
            background: Optional background tasks
            
        Returns:
            Dictionary containing complete processing results and response
        """
        logger.info(f"🚀 Starting stateful pipeline for user {user_id}")
        logger.info(f"📝 Query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        logger.info(f"📊 Conversation history: {len(conversation_history) if conversation_history else 0} messages")
        
        # Create initial state
        initial_state = create_initial_pipeline_state(
            query, 
            user_id, 
            conversation_id, 
            message_id, 
            conversation_history, 
            user_gender, 
            db, 
            background
        )
        logger.info(f"🔧 Initial pipeline state created with user_gender: {user_gender}")
        
        # Language detection will happen within the query validation node
        logger.info("🌍 Language detection will be handled by query validation node")
        
        try:
            # Execute the pipeline
            start_time = time.time()
            logger.info("⚡ Executing LangGraph pipeline...")
            final_state = await self.graph.ainvoke(initial_state)
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Pipeline completed in {processing_time:.2f}s")
            logger.info(f"   📋 Steps completed: {final_state.get('processing_steps_completed', [])}")
            logger.info(f"   🤖 LLM calls made: {final_state.get('llm_calls_made', 0)}")
            logger.info(f"   ❌ Errors encountered: {len(final_state.get('errors', []))}")
            
            if final_state.get('errors'):
                logger.warning(f"   ⚠️  Error details: {final_state.get('errors', [])}")
            
            # Extract results
            results = self._extract_results(final_state, processing_time)
            logger.info(f"📤 Pipeline results extracted: response length={len(results.get('response', ''))} chars, confidence={results.get('response_confidence', 0):.2f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            logger.error(f"   🔍 Error type: {type(e).__name__}")
            logger.error(f"   📍 Error details: {str(e)}")
            return self._get_fallback_result(query, str(e))
    
    async def process_query_stream(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_gender: Optional[str] = None,
        db: Optional[Session] = None,
        background: Optional[BackgroundTasks] = None,
    ):
        """
        Like process_query() but streams the final response token by token.

        Yields str tokens as they arrive from the LLM.  After the generator is
        exhausted, state["generated_content"] contains the full response text
        (already set by EmpathyNode.execute_stream or the non-streaming fallback).

        The caller is responsible for saving the bot message to the database.
        """
        state = create_initial_pipeline_state(
            query, user_id, conversation_id, message_id,
            conversation_history, user_gender, db, background
        )

        # Fast local nodes first (no LLM cost)
        state = await self.emotion_detection_node.execute(state)
        # Single LLM call for classification
        state = await self.unified_analysis_node.execute(state)

        route = self._route_after_unified_analysis(state)
        logger.info(f"[stream] route → {route}")

        if route == "rag_enhancement" and self.rag_enhancement_node:
            state = await self.rag_enhancement_node.execute(state)
            async for token in self.empathy_node.execute_stream(state):
                yield token

        elif route == "empathy":
            async for token in self.empathy_node.execute_stream(state):
                yield token

        elif route == "crisis_alert":
            state = await self.crisis_alert_node.execute(state)
            state = await self.generate_response_node.execute(state)
            full = state.get("generated_content", "I'm here to support you.")
            yield full

        else:  # generate_response (greetings / off-topic)
            state = await self.generate_response_node.execute(state)
            full = state.get("generated_content", "I'm here to support you.")
            yield full

    def _extract_results(self, state: StatefulPipelineState, processing_time: float) -> Dict[str, Any]:
        """Extract and format results from final state."""
        return {
            "response": state.get("generated_content", "I'm here to support you."),
            "response_confidence": state.get("response_confidence", 0.0),
            "response_reason": state.get("response_reason", ""),
            "processing_metadata": [
                {
                    "step": meta.step_name,
                    "confidence": meta.confidence_score,
                    "reasoning": meta.reasoning,
                    "keywords": meta.keywords,
                    "processing_time": meta.processing_time
                }
                for meta in state.get("processing_metadata", [])
            ],
            "query_validation": state.get("query_validation"),
            "crisis_assessment": state.get("crisis_assessment"),
            "emotion_detection": state.get("emotion_detection"),
            "query_evaluation": state.get("query_evaluation"),
            "cultural_context_applied": state.get("cultural_context_applied", []),
            "processing_time": processing_time,
            "llm_calls_made": state.get("llm_calls_made", 0),
            "errors": state.get("errors", [])
        }
    
    def _get_fallback_result(self, query: str, error: str) -> Dict[str, Any]:
        """Get fallback result when pipeline fails."""
        return {
            "response": "I'm here to support you. How can I help you today?",
            "response_confidence": 0.0,
            "response_reason": f"Pipeline error: {error}",
            "processing_metadata": [],
            "errors": [error],
            "processing_time": 0.0,
            "llm_calls_made": 0
        }

    # Node wrapper methods for LangGraph compatibility
    async def _query_validation_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("🔍 [PIPELINE] Executing Query Validation Node")
        result = await self.query_validation_node.execute(state)
        logger.info("✅ [PIPELINE] Query Validation Node completed")
        return result
    
    async def _crisis_detection_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("🚨 [PIPELINE] Executing Crisis Detection Node")
        result = await self.crisis_detection_node.execute(state)
        logger.info("✅ [PIPELINE] Crisis Detection Node completed")
        return result

    async def _unified_analysis_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("[PIPELINE] Executing Unified Analysis Node")
        result = await self.unified_analysis_node.execute(state)
        logger.info("[PIPELINE] Unified Analysis Node completed")
        return result

    async def _emotion_detection_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("😊 [PIPELINE] Executing Emotion Detection Node")
        result = await self.emotion_detection_node.execute(state)
        logger.info("✅ [PIPELINE] Emotion Detection Node completed")
        return result
    
    async def _query_evaluation_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("🎯 [PIPELINE] Executing Query Evaluation Node")
        result = await self.query_evaluation_node.execute(state)
        logger.info("✅ [PIPELINE] Query Evaluation Node completed")
        return result
    
    async def _rag_enhancement_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("🔍 [PIPELINE] Executing RAG Enhancement Node")
        result = await self.rag_enhancement_node.execute(state)
        logger.info("✅ [PIPELINE] RAG Enhancement Node completed")
        return result
    
    async def _elaboration_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("💬 [PIPELINE] Executing Elaboration Node")
        result = await self.elaboration_node.execute(state)
        logger.info("✅ [PIPELINE] Elaboration Node completed")
        return result
    
    async def _empathy_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("💝 [PIPELINE] Executing Empathy Node")
        result = await self.empathy_node.execute(state)
        logger.info("✅ [PIPELINE] Empathy Node completed")
        return result
    
    async def _clarification_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("❓ [PIPELINE] Executing Clarification Node")
        result = await self.clarification_node.execute(state)
        logger.info("✅ [PIPELINE] Clarification Node completed")
        return result
    
    async def _suggestion_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("💡 [PIPELINE] Executing Suggestion Node")
        result = await self.suggestion_node.execute(state)
        logger.info("✅ [PIPELINE] Suggestion Node completed")
        return result
    
    async def _guidance_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("🧭 [PIPELINE] Executing Guidance Node")
        result = await self.guidance_node.execute(state)
        logger.info("✅ [PIPELINE] Guidance Node completed")
        return result
    
    async def _idle_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("😴 [PIPELINE] Executing Idle Node")
        result = await self.idle_node.execute(state)
        logger.info("✅ [PIPELINE] Idle Node completed")
        return result
    
    async def _crisis_alert_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("🚨 [PIPELINE] Executing Crisis Alert Node")
        result = await self.crisis_alert_node.execute(state)
        logger.info("✅ [PIPELINE] Crisis Alert Node completed")
        return result
    
    async def _generate_response_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("📝 [PIPELINE] Executing Generate Response Node")
        result = await self.generate_response_node.execute(state)
        logger.info("✅ [PIPELINE] Generate Response Node completed")
        return result


def initialize_stateful_pipeline(llm_provider=None, db: Optional[Session] = None, background: Optional[BackgroundTasks] = None) -> StatefulMentalHealthPipeline:
    """Initialize the stateful mental health pipeline."""
    return StatefulMentalHealthPipeline(llm_provider, db=db, background=background)

