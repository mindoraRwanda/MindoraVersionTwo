"""
Stateful Mental Health Pipeline Orchestrator

This module implements the main LangGraph orchestrator for the unified mental health
chatbot pipeline with comprehensive state management and explainability.
"""

from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import asyncio
import time
import logging

from .pipeline_state import (
    StatefulPipelineState, ResponseStrategy, CrisisSeverity, QueryType,
    create_initial_pipeline_state, add_processing_metadata, increment_llm_calls, add_error
)
from .pipeline_nodes import (
    QueryValidationNode, CrisisDetectionNode, EmotionDetectionNode,
    QueryEvaluationNode, ElaborationNode, EmpathyNode, ClarificationNode,
    SuggestionNode, GuidanceNode, IdleNode, CrisisAlertNode, GenerateResponseNode
)
from .rag_enhancement_node import RAGEnhancementNode
from ..settings import settings
from ..prompts.cultural_context_prompts import CulturalContextPrompts

# Import ML emotion classifier
from .emotion.text_emotion_classifier import TextEmotionClassifier

# Import metrics trackers
from .metrics import EmotionMetricsTracker, PipelineMetricsTracker

logger = logging.getLogger(__name__)


class StatefulMentalHealthPipeline:
    """
    Main orchestrator for the stateful LangGraph mental health pipeline.
    
    This class manages the complete workflow from query validation through
    response generation, with explainability and state tracking.
    """

    def __init__(self, llm_provider=None, rag_service=None):
        """Initialize the stateful pipeline with LLM provider and RAG service."""
        import time
        pipeline_start = time.time()
        
        logger.info("🔧 [PIPELINE] Initializing StatefulMentalHealthPipeline...")
        self.llm_provider = llm_provider
        self.rag_service = rag_service
        self.cultural_prompts = CulturalContextPrompts()
        logger.info("📚 [PIPELINE] Cultural context prompts loaded")
        
        # Initialize ML emotion classifier
        logger.info("🧠 [PIPELINE] Initializing ML emotion classifier (DistilRoBERTa)...")
        emotion_start = time.time()
        try:
            self.ml_emotion_classifier = TextEmotionClassifier()
            emotion_time = time.time() - emotion_start
            logger.info(f"✅ [PIPELINE] ML emotion classifier initialized successfully in {emotion_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ [PIPELINE] Failed to initialize ML emotion classifier: {e}")
            self.ml_emotion_classifier = None
            logger.warning("⚠️ [PIPELINE] Falling back to LLM-based emotion detection")
        
        # Initialize metrics trackers
        logger.info("📊 [PIPELINE] Initializing metrics trackers...")
        self.emotion_metrics = EmotionMetricsTracker()
        self.pipeline_metrics = PipelineMetricsTracker()
        logger.info("✅ [PIPELINE] Metrics trackers initialized")
        
        # Initialize node instances
        logger.info("🏗️  [PIPELINE] Initializing pipeline nodes...")
        self.query_validation_node = QueryValidationNode(self.llm_provider)
        self.crisis_detection_node = CrisisDetectionNode(self.llm_provider)
        # Pass ML classifier to emotion detection node
        self.emotion_detection_node = EmotionDetectionNode(
            llm_provider=self.llm_provider,
            ml_classifier=self.ml_emotion_classifier
        )
        self.query_evaluation_node = QueryEvaluationNode(self.llm_provider)
        self.elaboration_node = ElaborationNode(self.llm_provider)
        self.empathy_node = EmpathyNode(self.llm_provider)
        self.clarification_node = ClarificationNode(self.llm_provider)
        self.suggestion_node = SuggestionNode(self.llm_provider)
        self.guidance_node = GuidanceNode(self.llm_provider)
        self.idle_node = IdleNode(self.llm_provider)
        self.crisis_alert_node = CrisisAlertNode(self.llm_provider)
        self.generate_response_node = GenerateResponseNode(self.llm_provider)
        self.idle_node = IdleNode(self.llm_provider)
        self.crisis_alert_node = CrisisAlertNode(self.llm_provider)
        self.generate_response_node = GenerateResponseNode(self.llm_provider)
        
        # Initialize RAG enhancement node if RAG service is available
        if self.rag_service:
            self.rag_enhancement_node = RAGEnhancementNode(self.rag_service)
            logger.info("🔍 [PIPELINE] RAG enhancement node initialized")
        else:
            self.rag_enhancement_node = None
            logger.info("⚠️ [PIPELINE] RAG service not available, skipping RAG enhancement node")
        
        logger.info("✅ [PIPELINE] All pipeline nodes initialized")
        
        logger.info("🕸️  [PIPELINE] Building LangGraph workflow...")
        self.graph = self._build_pipeline_graph()
        
        pipeline_time = time.time() - pipeline_start
        logger.info(f"✅ [PIPELINE] StatefulMentalHealthPipeline initialization complete in {pipeline_time:.2f}s")
        
    def _build_pipeline_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph with all nodes and routing logic."""
        logger.info("🏗️  Building LangGraph StateGraph...")
        
        # Initialize the graph
        workflow = StateGraph(StatefulPipelineState)
        logger.info("📊 StateGraph initialized with StatefulPipelineState")
        
        # Add all nodes
        logger.info("🔗 Adding nodes to workflow...")
        workflow.add_node("query_validation", self._query_validation_node)
        workflow.add_node("crisis_detection", self._crisis_detection_node)
        workflow.add_node("emotion_detection", self._emotion_detection_node)
        workflow.add_node("query_evaluation", self._query_evaluation_node)
        
        # Add RAG enhancement node if available
        if self.rag_enhancement_node:
            workflow.add_node("rag_enhancement", self._rag_enhancement_node)
            logger.info("🔍 RAG enhancement node added to workflow")
        workflow.add_node("elaboration", self._elaboration_node)
        workflow.add_node("empathy", self._empathy_node)
        workflow.add_node("clarification", self._clarification_node)
        workflow.add_node("suggestion", self._suggestion_node)
        workflow.add_node("guidance", self._guidance_node)
        workflow.add_node("idle", self._idle_node)
        workflow.add_node("crisis_alert", self._crisis_alert_node)
        workflow.add_node("generate_response", self._generate_response_node)
        logger.info("✅ All 12 nodes added to workflow")
        
        # Set entry point
        workflow.set_entry_point("query_validation")
        logger.info("🚪 Entry point set to 'query_validation'")
        
        # Add conditional routing
        logger.info("🔀 Adding conditional routing edges...")
        workflow.add_conditional_edges(
            "query_validation",
            self._route_after_validation,
            {
                "crisis_detection": "crisis_detection",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "crisis_detection",
            self._route_after_crisis,
            {
                "emotion_detection": "emotion_detection",
                "crisis_alert": "crisis_alert",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "emotion_detection",
            self._route_after_emotion,
            {
                "query_evaluation": "query_evaluation"
            }
        )
        
        # Build routing options for query evaluation
        evaluation_routes = {
            "elaboration": "elaboration",
            "empathy": "empathy",
            "clarification": "clarification",
            "suggestion": "suggestion",
            "guidance": "guidance",
            "idle": "idle"
        }
        
        # Add RAG enhancement route only if available
        if self.rag_enhancement_node:
            evaluation_routes["rag_enhancement"] = "rag_enhancement"
        
        workflow.add_conditional_edges(
            "query_evaluation",
            self._route_after_evaluation,
            evaluation_routes
        )
        
        # Add RAG enhancement routing if available
        if self.rag_enhancement_node:
            workflow.add_conditional_edges(
                "rag_enhancement",
                self._route_after_rag_enhancement,
                {
                    "elaboration": "elaboration",
                    "empathy": "empathy",
                    "clarification": "clarification",
                    "suggestion": "suggestion",
                    "guidance": "guidance",
                    "idle": "idle"
                }
            )
        logger.info("✅ Conditional routing edges added")
        
        # All response nodes lead to generate_response
        logger.info("🔗 Adding response node edges...")
        for node in ["elaboration", "empathy", "clarification", "suggestion", "guidance", "idle", "crisis_alert"]:
            workflow.add_edge(node, "generate_response")
        
        workflow.add_edge("generate_response", END)
        logger.info("✅ All edges added")
        
        logger.info("⚡ Compiling workflow...")
        compiled_workflow = workflow.compile()
        logger.info("✅ LangGraph workflow compiled successfully")
        return compiled_workflow
    
    def _route_after_validation(self, state: StatefulPipelineState) -> str:
        """Route after query validation based on results."""
        validation = state.get("query_validation")
        if not validation:
            logger.info("🔄 No validation result found, ending pipeline")
            return "end"
        
        if validation.is_random:
            logger.info(f"🔄 Query marked as random (confidence: {validation.query_confidence:.2f}), ending pipeline")
            return "end"
        
        logger.info(f"✅ Query validated as mental health related (confidence: {validation.query_confidence:.2f}), proceeding to crisis detection")
        return "crisis_detection"
    
    def _route_after_crisis(self, state: StatefulPipelineState) -> str:
        """Route after crisis detection based on severity."""
        crisis = state.get("crisis_assessment")
        if not crisis:
            logger.info("🔄 No crisis assessment found, proceeding to emotion detection")
            return "emotion_detection"
        
        severity = crisis.crisis_severity
        confidence = crisis.crisis_confidence
        
        if severity in [CrisisSeverity.SEVERE, CrisisSeverity.HIGH]:
            logger.warning(f"🚨 CRISIS DETECTED: {severity.value} (confidence: {confidence:.2f}), routing to crisis alert")
            return "crisis_alert"
        
        logger.info(f"✅ Crisis assessment completed: {severity.value} (confidence: {confidence:.2f}), proceeding to emotion detection")
        return "emotion_detection"
    
    def _route_after_emotion(self, state: StatefulPipelineState) -> str:
        """Route after emotion detection."""
        emotion = state.get("emotion_detection")
        if emotion:
            logger.info(f"✅ Emotion detection completed: {emotion.selected_emotion} (confidence: {emotion.emotion_confidence:.2f}), proceeding to query evaluation")
        else:
            logger.info("🔄 No emotion detection result, proceeding to query evaluation")
        return "query_evaluation"
    
    def _route_after_evaluation(self, state: StatefulPipelineState) -> str:
        """Route after query evaluation based on strategy."""
        evaluation = state.get("query_evaluation")
        if not evaluation:
            logger.info("🔄 No evaluation found, routing to idle")
            return "idle"
        
        # Route to RAG enhancement first if available
        if self.rag_enhancement_node:
            logger.info("🔍 Routing to RAG enhancement first")
            return "rag_enhancement"
        
        # Direct routing if no RAG enhancement
        strategy = evaluation.evaluation_type
        
        # Map strategy values to node names
        strategy_to_node = {
            "GIVE_EMPATHY": "empathy",
            "AWAIT_ELABORATION": "elaboration", 
            "AWAIT_CLARIFICATION": "clarification",
            "GIVE_SUGGESTION": "suggestion",
            "GIVE_GUIDANCE": "guidance",
            "IDLE": "idle"
        }
        
        target_node = strategy_to_node.get(strategy.value, "idle")
        logger.info(f"🎯 Routing decision: {strategy.value} -> {target_node}")
        return target_node
    
    def _route_after_rag_enhancement(self, state: StatefulPipelineState) -> str:
        """Route after RAG enhancement based on original evaluation strategy."""
        evaluation = state.get("query_evaluation")
        if not evaluation:
            logger.info("🔄 No evaluation found after RAG enhancement, routing to idle")
            return "idle"
        
        strategy = evaluation.evaluation_type
        
        # Map strategy values to node names
        strategy_to_node = {
            "GIVE_EMPATHY": "empathy",
            "AWAIT_ELABORATION": "elaboration", 
            "AWAIT_CLARIFICATION": "clarification",
            "GIVE_SUGGESTION": "suggestion",
            "GIVE_GUIDANCE": "guidance",
            "IDLE": "idle"
        }
        
        target_node = strategy_to_node.get(strategy.value, "idle")
        logger.info(f"🎯 Post-RAG routing decision: {strategy.value} -> {target_node}")
        return target_node
    
    async def process_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_gender: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete stateful pipeline.
        
        Args:
            query: User's input query
            user_id: Optional user identifier
            conversation_history: Optional conversation history
            user_gender: Optional user gender for cultural context (male, female, other, prefer_not_to_say)
            
        Returns:
            Dictionary containing complete processing results and response
        """
        logger.info(f"🚀 Starting stateful pipeline for user {user_id}")
        logger.info(f"📝 Query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        logger.info(f"📊 Conversation history: {len(conversation_history) if conversation_history else 0} messages")
        
        # Create initial state
        initial_state = create_initial_pipeline_state(query, user_id, conversation_history, user_gender)
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
                logger.warning(f"   ⚠️  Error  details: {final_state.get('errors', [])}")
            
            # Extract results
            results = self._extract_results(final_state, processing_time)
            logger.info(f"📤 Pipeline results extracted: response length={len(results.get('response', ''))} chars, confidence={results.get('response_confidence', 0):.2f}")
            
            # Track pipeline metrics
            try:
                emotion_det = final_state.get("emotion_detection")
                crisis_assess = final_state.get("crisis_assessment")
                query_eval = final_state.get("query_evaluation")
                
                # Safely extract strategy (it's called evaluation_type in QueryEvaluation)
                strategy_used = "unknown"
                if query_eval:
                    if hasattr(query_eval, 'evaluation_type'):
                        strategy_used = query_eval.evaluation_type.value if hasattr(query_eval.evaluation_type, 'value') else str(query_eval.evaluation_type)
                    elif hasattr(query_eval, 'strategy'):
                        strategy_used = query_eval.strategy.value if hasattr(query_eval.strategy, 'value') else str(query_eval.strategy)
                
                # Safely extract crisis severity
                crisis_severity = "none"
                if crisis_assess:
                    if hasattr(crisis_assess, 'crisis_severity'):
                        crisis_severity = crisis_assess.crisis_severity.value if hasattr(crisis_assess.crisis_severity, 'value') else str(crisis_assess.crisis_severity)
                    elif hasattr(crisis_assess, 'severity'):
                        crisis_severity = crisis_assess.severity.value if hasattr(crisis_assess.severity, 'value') else str(crisis_assess.severity)
                
                # Safely extract emotion
                detected_emotion = "unknown"
                if emotion_det:
                    if hasattr(emotion_det, 'selected_emotion'):
                        detected_emotion = emotion_det.selected_emotion.value if hasattr(emotion_det.selected_emotion, 'value') else str(emotion_det.selected_emotion)
                    elif hasattr(emotion_det, 'primary_emotion'):
                        detected_emotion = emotion_det.primary_emotion.value if hasattr(emotion_det.primary_emotion, 'value') else str(emotion_det.primary_emotion)
                
                self.pipeline_metrics.track(
                    user_id=str(user_id or "unknown"),
                    message_id=str(final_state.get("message_id", "unknown")),
                    query_length=len(query),
                    total_time_ms=processing_time * 1000,
                    llm_calls=final_state.get("llm_calls_made", 0),
                    errors=len(final_state.get("errors", [])),
                    steps_completed=final_state.get("processing_steps_completed", []),
                    response_length=len(results.get("response", "")),
                    response_confidence=results.get("response_confidence", 0.0),
                    strategy=strategy_used,
                    crisis_severity=crisis_severity,
                    detected_emotion=detected_emotion
                )
                logger.info(f"📊 Pipeline metrics tracked: strategy={strategy_used}, crisis={crisis_severity}, emotion={detected_emotion}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to track pipeline metrics: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            logger.error(f"   🔍 Error type: {type(e).__name__}")
            logger.error(f"   📍 Error details: {str(e)}")
            return self._get_fallback_result(query, str(e))
    
    def _extract_results(self, state: StatefulPipelineState, processing_time: float) -> Dict[str, Any]:
        """Extract and format results from final state."""
        return {
            "response": state.get("generated_content", "I'm here to support you."),
            "should_chunk": state.get("should_chunk", False),
            "response_chunks": state.get("response_chunks", []),
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
    
    async def _emotion_detection_node(self, state: StatefulPipelineState) -> StatefulPipelineState:
        logger.info("😊 [PIPELINE] Executing Emotion Detection Node")
        start_time = time.time()
        result = await self.emotion_detection_node.execute(state)
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Track emotion metrics
        if result.get("emotion_detection"):
            emotion_det = result["emotion_detection"]
            try:
                self.emotion_metrics.track(
                    user_id=str(result.get("user_id", "unknown")),
                    message_id=str(result.get("message_id", "unknown")),
                    text=result.get("user_query", ""),
                    primary_emotion=emotion_det.selected_emotion,
                    confidence=emotion_det.emotion_confidence,
                    intensity=result.get("emotion_intensity", "unknown"),
                    processing_time_ms=processing_time_ms,
                    secondary_emotions=result.get("emotion_secondary", {}),
                    cultural_context=result.get("emotion_cultural_context")
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to track emotion metrics: {e}")
        
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


def initialize_stateful_pipeline(llm_provider=None) -> StatefulMentalHealthPipeline:
    """Initialize the stateful mental health pipeline."""
    return StatefulMentalHealthPipeline(llm_provider)
