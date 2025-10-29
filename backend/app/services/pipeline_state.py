"""
Pipeline State Management for Stateful LangGraph Mental Health Pipeline

This module defines the state management system for the unified mental health
chatbot pipeline, providing comprehensive state tracking and explainability.
"""

from typing import Dict, List, Any, Optional, TypedDict, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


class QueryType(Enum):
    """Types of user queries for classification."""
    MENTAL_HEALTH = "mental_health"
    CRISIS = "crisis"
    RANDOM = "random"
    UNCLEAR = "unclear"


class CrisisSeverity(Enum):
    """Crisis severity levels for intervention protocols."""
    SEVERE = "severe"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class ResponseStrategy(Enum):
    """Response strategies for different user needs."""
    GIVE_EMPATHY = "GIVE_EMPATHY"
    AWAIT_ELABORATION = "AWAIT_ELABORATION"
    AWAIT_CLARIFICATION = "AWAIT_CLARIFICATION"
    GIVE_SUGGESTION = "GIVE_SUGGESTION"
    GIVE_GUIDANCE = "GIVE_GUIDANCE"
    IDLE = "IDLE"


class ProcessingMetadata(BaseModel):
    """Metadata for processing steps with explainability."""
    step_name: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    keywords: List[str] = []
    processing_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    errors: List[str] = []


class QueryValidationResult(BaseModel):
    """Results from query validation node."""
    query_confidence: float = Field(ge=0.0, le=1.0)
    query_keywords: List[str] = []
    query_reason: str
    is_random: bool
    query_type: QueryType = QueryType.UNCLEAR


class CrisisAssessment(BaseModel):
    """Results from crisis detection node."""
    crisis_confidence: float = Field(ge=0.0, le=1.0)
    crisis_keywords: List[str] = []
    crisis_reason: str
    crisis_severity: CrisisSeverity = CrisisSeverity.NONE


class EmotionDetection(BaseModel):
    """Results from emotion detection node."""
    emotions: Dict[str, float] = {}
    emotion_keywords: List[str] = []
    emotion_reason: str
    selected_emotion: str = "neutral"
    emotion_confidence: float = Field(ge=0.0, le=1.0)


class QueryEvaluation(BaseModel):
    """Results from query evaluation node."""
    evaluation_confidence: float = Field(ge=0.0, le=1.0)
    evaluation_reason: str
    evaluation_keywords: List[str] = []
    evaluation_type: ResponseStrategy = ResponseStrategy.IDLE


class StatefulPipelineState(TypedDict):
    """Main state for the LangGraph pipeline."""
    # Input data
    user_query: str
    user_id: Optional[str]
    conversation_history: List[Dict[str, Any]]
    
    # Processing results
    query_validation: Optional[QueryValidationResult]
    crisis_assessment: Optional[CrisisAssessment]
    emotion_detection: Optional[EmotionDetection]
    query_evaluation: Optional[QueryEvaluation]
    
    # Response generation
    generated_content: str
    response_confidence: float
    response_reason: str
    
    # Metadata and explainability
    processing_metadata: List[ProcessingMetadata]
    processing_steps_completed: List[str]
    llm_calls_made: int
    errors: List[str]
    
    # Pipeline control
    should_continue: bool
    current_node: str
    next_node: Optional[str]
    
    # Cultural context
    detected_language: Optional[str]
    cultural_context_applied: List[str]
    gender_aware_addressing: Optional[str]
    user_gender: Optional[str]
    cultural_context_confidence: float
    cultural_adaptations: List[Dict[str, Any]]
    
    # RAG enhancement
    retrieved_knowledge: List[Dict[str, Any]]
    knowledge_context: str
    rag_enhancement_applied: bool
    rag_relevance_score: float


def create_initial_pipeline_state(
    query: str,
    user_id: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    user_gender: Optional[str] = None
) -> StatefulPipelineState:
    """Create initial state for the pipeline."""
    return {
        "user_query": query,
        "user_id": user_id,
        "conversation_history": conversation_history or [],
        "query_validation": None,
        "crisis_assessment": None,
        "emotion_detection": None,
        "query_evaluation": None,
        "generated_content": "",
        "response_confidence": 0.0,
        "response_reason": "",
        "processing_metadata": [],
        "processing_steps_completed": [],
        "llm_calls_made": 0,
        "errors": [],
        "should_continue": True,
        "current_node": "query_validation",
        "next_node": None,
        "detected_language": None,
        "cultural_context_applied": [],
        "gender_aware_addressing": None,
        "user_gender": user_gender,
        "cultural_context_confidence": 0.0,
        "cultural_adaptations": [],
        "retrieved_knowledge": [],
        "knowledge_context": "",
        "rag_enhancement_applied": False,
        "rag_relevance_score": 0.0
    }


def add_processing_metadata(
    state: StatefulPipelineState,
    step_name: str,
    confidence: float,
    reasoning: str,
    keywords: List[str] = None,
    processing_time: float = 0.0,
    errors: List[str] = None
) -> StatefulPipelineState:
    """Add processing metadata to state."""
    metadata = ProcessingMetadata(
        step_name=step_name,
        confidence_score=confidence,
        reasoning=reasoning,
        keywords=keywords or [],
        processing_time=processing_time,
        errors=errors or []
    )
    
    if "processing_metadata" not in state:
        state["processing_metadata"] = []
    state["processing_metadata"].append(metadata)
    
    if "processing_steps_completed" not in state:
        state["processing_steps_completed"] = []
    state["processing_steps_completed"].append(step_name)
    
    return state


def increment_llm_calls(state: StatefulPipelineState) -> StatefulPipelineState:
    """Increment LLM calls counter."""
    state["llm_calls_made"] = state.get("llm_calls_made", 0) + 1
    return state


def add_error(state: StatefulPipelineState, error: str) -> StatefulPipelineState:
    """Add error to state."""
    if "errors" not in state:
        state["errors"] = []
    state["errors"].append(error)
    return state


def add_cultural_context(
    state: StatefulPipelineState,
    context_type: str,
    context_value: str,
    confidence: float = 1.0,
    reasoning: str = ""
) -> StatefulPipelineState:
    """Add cultural context application to state."""
    if "cultural_context_applied" not in state:
        state["cultural_context_applied"] = []
    state["cultural_context_applied"].append(context_type)
    
    if "cultural_adaptations" not in state:
        state["cultural_adaptations"] = []
    
    adaptation = {
        "type": context_type,
        "value": context_value,
        "confidence": confidence,
        "reasoning": reasoning,
        "timestamp": datetime.now().isoformat()
    }
    state["cultural_adaptations"].append(adaptation)
    
    return state


def set_detected_language(
    state: StatefulPipelineState,
    language: str,
    confidence: float = 1.0
) -> StatefulPipelineState:
    """Set detected language in state."""
    state["detected_language"] = language
    state["cultural_context_confidence"] = confidence
    
    # Add cultural context tracking
    add_cultural_context(
        state,
        "language_detection",
        language,
        confidence,
        f"Detected language: {language} with confidence {confidence:.2f}"
    )
    
    return state


def set_gender_aware_addressing(
    state: StatefulPipelineState,
    addressing: str,
    confidence: float = 1.0
) -> StatefulPipelineState:
    """Set gender-aware addressing in state."""
    state["gender_aware_addressing"] = addressing
    
    # Add cultural context tracking
    add_cultural_context(
        state,
        "gender_addressing",
        addressing,
        confidence,
        f"Applied gender-aware addressing: {addressing}"
    )
    
    return state


def get_cultural_context_summary(state: StatefulPipelineState) -> Dict[str, Any]:
    """Get summary of cultural context applications."""
    return {
        "detected_language": state.get("detected_language"),
        "user_gender": state.get("user_gender"),
        "gender_aware_addressing": state.get("gender_aware_addressing"),
        "cultural_context_confidence": state.get("cultural_context_confidence", 0.0),
        "contexts_applied": state.get("cultural_context_applied", []),
        "adaptations_count": len(state.get("cultural_adaptations", [])),
        "recent_adaptations": state.get("cultural_adaptations", [])[-3:] if state.get("cultural_adaptations") else []
    }
