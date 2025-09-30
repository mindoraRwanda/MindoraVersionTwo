"""
LangGraph state definitions for the query validation workflow.

This module defines the state structures used in the LangGraph workflow
for query validation and processing.
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class QueryType(Enum):
    """Types of queries that can be identified"""
    MENTAL_SUPPORT = "mental_support"
    RANDOM_QUESTION = "random_question"
    UNCLEAR = "unclear"
    CRISIS = "crisis"


class CrisisSeverity(Enum):
    """Severity levels for crisis situations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RoutingPriority(Enum):
    """Priority levels for query routing"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QueryClassification(TypedDict):
    """Structure for query classification results"""
    query_type: QueryType
    confidence: float
    reasoning: str
    keywords_found: List[str]
    is_crisis: bool
    crisis_severity: Optional[CrisisSeverity]
    requires_human_intervention: bool


class CrisisAssessment(TypedDict):
    """Structure for crisis assessment results"""
    is_crisis: bool
    confidence: float
    crisis_indicators: List[str]
    severity: CrisisSeverity
    recommended_action: str
    immediate_risks: List[str]


class QuerySuggestions(TypedDict):
    """Structure for query handling suggestions"""
    suggestions: List[str]
    routing_priority: RoutingPriority
    requires_human_intervention: bool
    follow_up_questions: List[str]
    next_best_action: str


class ConversationContext(TypedDict):
    """Structure for conversation context"""
    conversation_type: str  # new, ongoing, crisis_followup, progress_check, topic_shift
    emotional_progression: str  # improving, stable, worsening, unclear
    key_themes: List[str]
    user_preferences: List[str]
    suggested_focus: str
    memory_cues: List[str]


class MemoryEntry(TypedDict):
    """Structure for memory entries"""
    category: str  # personal, emotional, progress, resource, communication
    content: str
    importance: str  # high, medium, low
    timeframe: str  # session, short_term, long_term


class ConversationMemory(TypedDict):
    """Structure for conversation memory management"""
    to_remember: List[MemoryEntry]
    to_forget: List[str]
    memory_summary: str
    privacy_notes: str


class EmotionDetection(TypedDict):
    """Structure for emotion detection results"""
    detected_emotion: str
    confidence: float
    emotion_score: Dict[str, float]  # All emotion scores
    reasoning: str
    intensity: str  # low, medium, high, very_high
    context_relevance: str  # very_low, low, medium, high, very_high


class UserProfile(TypedDict):
    """Structure for user profile information"""
    user_id: Optional[str]
    emotional_patterns: Dict[str, Any]
    communication_preferences: Dict[str, Any]
    previous_interactions: int
    last_interaction: Optional[datetime]
    crisis_history: List[Dict[str, Any]]


class QueryValidationState(TypedDict):
    """
    Main state for the query validation LangGraph workflow.

    This state inherits from TypedDict and contains all the information
    that flows through the LangGraph workflow.
    """
    # Input data
    query: str
    user_id: Optional[str]
    conversation_history: List[Dict[str, Any]]

    # Classification results
    classification: Optional[QueryClassification]
    crisis_assessment: Optional[CrisisAssessment]
    suggestions: Optional[QuerySuggestions]

    # Emotion detection
    emotion_detection: Optional[EmotionDetection]

    # Context analysis
    conversation_context: Optional[ConversationContext]
    memory_management: Optional[ConversationMemory]

    # User information
    user_profile: Optional[UserProfile]

    # Processing metadata
    processing_timestamp: datetime
    processing_steps_completed: List[str]
    errors: List[str]

    # Output data
    final_response: Optional[str]
    routing_decision: Optional[str]
    confidence_score: float

    # Workflow control
    should_continue: bool
    next_action: Optional[str]
    requires_escalation: bool
    should_proceed_to_conversation: bool
    should_proceed_to_conversation: bool


class WorkflowStep(Enum):
    """Steps in the query validation workflow"""
    INITIALIZE = "initialize"
    CLASSIFY_QUERY = "classify_query"
    ASSESS_CRISIS = "assess_crisis"
    DETECT_EMOTION = "detect_emotion"
    GENERATE_SUGGESTIONS = "generate_suggestions"
    ANALYZE_CONTEXT = "analyze_context"
    MANAGE_MEMORY = "manage_memory"
    MAKE_ROUTING_DECISION = "make_routing_decision"
    GENERATE_RESPONSE = "generate_response"
    COMPLETE = "complete"


class ProcessingMetadata(TypedDict):
    """Metadata for workflow processing"""
    workflow_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_steps: int
    completed_steps: int
    current_step: Optional[WorkflowStep]
    processing_time: float
    llm_calls_made: int
    errors_encountered: List[str]


class LangGraphConfig(TypedDict):
    """Configuration for the LangGraph workflow"""
    max_steps: int
    timeout_seconds: int
    llm_model: str
    temperature: float
    enable_memory: bool
    enable_crisis_detection: bool
    enable_context_analysis: bool
    crisis_threshold: float
    confidence_threshold: float


# Default configurations
DEFAULT_CONFIG: LangGraphConfig = {
    "max_steps": 10,
    "timeout_seconds": 30,
    "llm_model": "HuggingFaceTB/SmolLM3-3B",
    "temperature": 0.9,
    "enable_memory": True,
    "enable_crisis_detection": True,
    "enable_context_analysis": True,
    "crisis_threshold": 0.7,
    "confidence_threshold": 0.6
}


def create_initial_state(
    query: str,
    user_id: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> QueryValidationState:
    """
    Create an initial state for the LangGraph workflow.

    Args:
        query: The user's query string
        user_id: Optional user identifier
        conversation_history: Optional conversation history

    Returns:
        Initial QueryValidationState
    """
    return QueryValidationState(
        # Input data
        query=query,
        user_id=user_id,
        conversation_history=conversation_history or [],

        # Classification results (initially None)
        classification=None,
        crisis_assessment=None,
        suggestions=None,

        # Emotion detection (initially None)
        emotion_detection=None,

        # Context analysis (initially None)
        conversation_context=None,
        memory_management=None,

        # User information (initially None)
        user_profile=None,

        # Processing metadata
        processing_timestamp=datetime.now(),
        processing_steps_completed=[],
        errors=[],

        # Output data (initially None)
        final_response=None,
        routing_decision=None,
        confidence_score=0.0,

        # Workflow control
        should_continue=True,
        next_action=None,
        requires_escalation=False,
        should_proceed_to_conversation=False
    )


def update_state_with_classification(
    state: QueryValidationState,
    classification: QueryClassification
) -> QueryValidationState:
    """
    Update state with query classification results.

    Args:
        state: Current state
        classification: Classification results

    Returns:
        Updated state
    """
    state["classification"] = classification
    state["processing_steps_completed"].append("classify_query")
    state["confidence_score"] = classification["confidence"]
    return state


def update_state_with_crisis_assessment(
    state: QueryValidationState,
    crisis_assessment: CrisisAssessment
) -> QueryValidationState:
    """
    Update state with crisis assessment results.

    Args:
        state: Current state
        crisis_assessment: Crisis assessment results

    Returns:
        Updated state
    """
    state["crisis_assessment"] = crisis_assessment
    state["processing_steps_completed"].append("assess_crisis")

    if crisis_assessment["is_crisis"]:
        state["requires_escalation"] = True
        state["routing_decision"] = "crisis_intervention"

    return state


def update_state_with_suggestions(
    state: QueryValidationState,
    suggestions: QuerySuggestions
) -> QueryValidationState:
    """
    Update state with query handling suggestions.

    Args:
        state: Current state
        suggestions: Query suggestions

    Returns:
        Updated state
    """
    state["suggestions"] = suggestions
    state["processing_steps_completed"].append("generate_suggestions")
    state["routing_decision"] = suggestions["next_best_action"]
    return state


def mark_workflow_complete(
    state: QueryValidationState,
    final_response: str,
    routing_decision: str
) -> QueryValidationState:
    """
    Mark the workflow as complete with final results.

    Args:
        state: Current state
        final_response: Final response to user
        routing_decision: Final routing decision

    Returns:
        Completed state
    """
    state["final_response"] = final_response
    state["routing_decision"] = routing_decision
    state["should_continue"] = False
    state["next_action"] = "complete"
    state["processing_steps_completed"].append("complete")
    return state