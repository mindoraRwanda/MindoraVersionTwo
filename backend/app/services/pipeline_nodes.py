"""
Pipeline Nodes for Stateful LangGraph Mental Health Pipeline

This module implements all the specialized nodes for the mental health pipeline,
including query validation, crisis detection, emotion detection, and response generation.
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import time
import logging
import json

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from sqlalchemy.orm import Session
from fastapi import BackgroundTasks

from .pipeline_state import (
    StatefulPipelineState, QueryValidationResult, CrisisAssessment,
    EmotionDetection, QueryEvaluation, ProcessingMetadata,
    QueryType, CrisisSeverity, ResponseStrategy, TherapeuticPhase,
    add_processing_metadata, increment_llm_calls, add_error,
    set_detected_language, add_cultural_context
)
from ..prompts.cultural_context_prompts import CulturalContextPrompts
from ..prompts.safety_prompts import SafetyPrompts
from ..prompts.response_approach_prompts import ResponseApproachPrompts
from .crisis_classifier import classify_crisis
from .safety_pipeline import log_crisis_and_notify
from .text_emotion_classifier import TextEmotionClassifier
from .llm_cultural_context import ConversationContextManager
from pydantic import BaseModel, Field


class QueryValidationOutput(BaseModel):
    """Pydantic model for query validation output."""
    confidence: float = Field(..., description="Confidence score (0-1) on whether the query is mental health related.")
    keywords: List[str] = Field(..., description="Keywords indicating mental health content.")
    reasoning: str = Field(..., description="Reasoning for the classification.")
    is_random: bool = Field(..., description="Whether the query is random/off-topic.")
    query_type: str = Field(..., description="Type of query, e.g., 'mental_health'.")
    cultural_indicators: List[str] = Field([], description="Cultural context considerations in the query.")


class UnifiedAnalysisOutput(BaseModel):
    """Single structured output that replaces QueryValidation + CrisisDetection + QueryEvaluation."""
    query_type: str = Field(..., description="One of: mental_health, greeting, casual, random, unclear")
    is_random: bool = Field(..., description="True if message is completely off-topic (not mental health)")
    query_confidence: float = Field(..., description="Confidence 0-1 that this is a mental health query")
    is_crisis: bool = Field(..., description="True ONLY for explicit suicidal ideation, active self-harm, or imminent danger")
    crisis_severity: str = Field(..., description="severe, high, medium, low, or none")
    crisis_confidence: float = Field(..., description="Crisis assessment confidence 0-1")
    crisis_keywords: List[str] = Field(default=[], description="Exact phrases that triggered the crisis flag")
    crisis_reason: str = Field(..., description="One sentence explaining the crisis assessment")

class CrisisDetectionOutput(BaseModel):
    """Pydantic model for crisis detection output."""
    is_crisis: bool = Field(..., description="Is crisis present? (true/false)")
    crisis_confidence: float = Field(..., description="Crisis confidence score (0-1)")
    crisis_keywords: List[str] = Field(..., description="Crisis keywords found (including cultural expressions)")
    crisis_reason: str = Field(..., description="Reasoning for assessment")
    crisis_severity: str = Field(..., description="crisis_severity level: severe, high, medium, low, none")

class EmotionDetectionOutput(BaseModel):
    """Pydantic model for emotion detection output."""
    emotions: Dict[str, float] = Field(..., description="Dictionary of emotions and their confidence scores.")
    keywords: List[str] = Field(..., description="List of emotion keywords (including cultural expressions)")
    reasoning: str = Field(..., description="Reasoning for emotion detection")
    selected_emotion: str = Field(..., description="The primary detected emotion.")
    confidence: float = Field(..., description="Confidence score for the selected emotion.")
    cultural_emotional_indicators: List[str] = Field([], description="Cultural emotional expression patterns")
    youth_emotional_patterns: List[str] = Field([], description="Youth-specific emotional language")

class QueryEvaluationOutput(BaseModel):
    """Pydantic model for query evaluation output."""
    confidence: float = Field(..., description="Confidence in the chosen strategy.")
    reasoning: str = Field(..., description="Reasoning for the strategy selection.")
    keywords: List[str] = Field(..., description="Keywords that influenced the decision.")
    strategy: str = Field(..., description="The selected response strategy.")
    cultural_considerations: List[str] = Field([], description="Cultural considerations for strategy selection.")
    cultural_appropriateness: str = Field("medium", description="Level of cultural appropriateness (high, medium, low).")

logger = logging.getLogger(__name__)


class BasePipelineNode(ABC):
    """Base class for all pipeline nodes with common functionality."""
    
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self.cultural_prompts = CulturalContextPrompts()
        self.safety_prompts = SafetyPrompts()
        self.response_prompts = ResponseApproachPrompts()
        # Language detection removed - default to English
    
    @abstractmethod
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Execute the node's processing logic."""
        pass
    
    @staticmethod
    def _build_session_context(history: List[Dict[str, Any]]) -> str:
        """Compress older conversation turns into a compact session narrative.

        Returns an empty string when history is short enough to inject verbatim.
        Only the last 4 messages are kept raw; everything before them is summarised here.
        """
        if len(history) <= 4:
            return ""

        older = history[:-4]
        lines = ["[Earlier in this session:]"]
        turn_num = 0
        for msg in older:
            role = str(msg.get("role", "user")).lower()
            text = (msg.get("text") or msg.get("content") or "").strip()
            if not text:
                continue
            if role == "user":
                turn_num += 1
                truncated = text[:150] + "..." if len(text) > 150 else text
                lines.append(f"  Turn {turn_num} (user): {truncated}")
            # Bot turns are implicit — omitting them halves token cost with minimal loss

        return "\n".join(lines) if len(lines) > 1 else ""

    @staticmethod
    def _compute_therapeutic_phase(exchange_count: int, history: List[Dict[str, Any]]) -> TherapeuticPhase:
        """Determine the current therapeutic phase from conversation progress.

        Primary signal is exchange count; a secondary distress signal slows
        advancement so the model doesn't rush through phases when the user is
        still processing heavy emotions.
        """
        # Detect sustained distress: if the last 3 user messages all contain
        # heavy-emotion markers, hold in the current phase one step longer.
        distress_words = {
            "hopeless", "hopelessness", "worthless", "can't go on", "end it",
            "no point", "hate myself", "give up", "can't cope", "breaking down",
        }
        recent_user_texts = [
            (msg.get("text") or msg.get("content") or "").lower()
            for msg in history[-6:]
            if str(msg.get("role", "")).lower() == "user"
        ]
        sustained_distress = (
            len(recent_user_texts) >= 2
            and sum(
                any(w in t for w in distress_words) for t in recent_user_texts
            ) >= 2
        )
        # Apply a -1 adjustment so phase advances one turn later under distress
        effective_count = max(1, exchange_count - (1 if sustained_distress else 0))

        if effective_count <= 2:
            return TherapeuticPhase.OPENING
        elif effective_count <= 5:
            return TherapeuticPhase.EXPLORING
        elif effective_count <= 8:
            return TherapeuticPhase.REFLECTING
        elif effective_count <= 12:
            return TherapeuticPhase.WORKING
        else:
            return TherapeuticPhase.CLOSING

    @staticmethod
    def _analyze_distress_trajectory(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Scan user turns for escalating or persistent distress patterns.

        Distress tiers (based on keyword matching):
          0 — neutral
          1 — mild      (sad, anxious, stressed, lonely…)
          2 — moderate  (hopeless, worthless, empty, can't cope…)
          3 — severe    (want to die, hurt myself, suicidal…)

        Returns a dict with:
          trajectory    : "escalating" | "persistent_high" | "stable" | "de-escalating"
          max_tier      : highest tier seen across all turns
          should_upgrade: True when the single-turn assessment should be raised
          turn_scores   : per-turn tier list (user messages only)
        """
        TIER_3 = {
            "want to die", "kill myself", "end my life", "commit suicide",
            "hurting myself", "cutting myself", "better off dead",
            "not worth living", "can't go on", "no reason to live",
        }
        TIER_2 = {
            "hopeless", "worthless", "empty inside", "feel empty", "numb",
            "can't cope", "no way out", "trapped", "breaking down",
            "want to disappear", "nobody cares", "hate myself",
        }
        TIER_1 = {
            "sad", "depressed", "anxious", "stressed", "lonely", "lost",
            "worried", "overwhelmed", "exhausted", "struggling", "suffering",
            "unhappy", "upset", "down", "low",
        }

        def _tier(text: str) -> int:
            t = text.lower()
            if any(w in t for w in TIER_3):
                return 3
            if any(w in t for w in TIER_2):
                return 2
            if any(w in t for w in TIER_1):
                return 1
            return 0

        user_turns = [
            msg for msg in history
            if str(msg.get("role", "")).lower() == "user"
        ]

        if len(user_turns) < 2:
            return {
                "trajectory": "stable",
                "max_tier": _tier(user_turns[0].get("text") or user_turns[0].get("content") or "") if user_turns else 0,
                "should_upgrade": False,
                "turn_scores": [],
            }

        scores = [
            _tier(msg.get("text") or msg.get("content") or "")
            for msg in user_turns
        ]
        max_tier = max(scores)
        recent = scores[-4:]  # analyse up to last 4 user turns

        # Escalating: each recent score >= previous AND final > first
        is_escalating = (
            len(recent) >= 3
            and all(recent[i] >= recent[i - 1] for i in range(1, len(recent)))
            and recent[-1] > recent[0]
            and recent[-1] >= 2
        )

        # Persistent high: average of recent scores >= 2, no turn below 1
        avg_recent = sum(recent) / len(recent)
        is_persistent = (
            len(recent) >= 3
            and avg_recent >= 2.0
            and min(recent) >= 1
        )

        if is_escalating:
            trajectory = "escalating"
        elif is_persistent:
            trajectory = "persistent_high"
        elif len(recent) >= 2 and recent[-1] < recent[0]:
            trajectory = "de-escalating"
        else:
            trajectory = "stable"

        # Only recommend an upgrade when there are enough turns to be confident
        should_upgrade = (
            trajectory in ("escalating", "persistent_high")
            and len(user_turns) >= 3
        )

        return {
            "trajectory": trajectory,
            "max_tier": max_tier,
            "should_upgrade": should_upgrade,
            "turn_scores": scores,
        }

    async def _call_llm(self, system_prompt: str, user_prompt: str, state: StatefulPipelineState, structured_output: Optional[Any] = None) -> Any:
        """Make LLM call with error handling and optional structured output."""
        try:
            if self.llm_provider is None:
                raise RuntimeError(
                    "LLM provider is not initialized. "
                    "Check the PROVIDER environment variable and API key settings."
                )
            logger.info(f"🤖 Making LLM call with {len(system_prompt)} char system prompt, {len(user_prompt)} char user prompt")
            system_message = SystemMessage(content=system_prompt)
            human_message = HumanMessage(content=user_prompt)

            # For response-generation calls (no structured schema), inject conversation
            # context so the model can give context-aware replies.
            # Strategy: compressed summary of older turns + last 4 messages verbatim.
            # Structured-output calls (classification/analysis) stay as [system, human]
            # to avoid polluting the JSON schema response.
            messages: List[Any] = [system_message]
            if not structured_output:
                history = state.get("conversation_history") or []

                # Inject compressed context for older turns (if any)
                session_ctx = self._build_session_context(history)
                if session_ctx:
                    messages.append(SystemMessage(content=session_ctx))

                # Inject last 4 messages verbatim for immediate context
                for turn in history[-4:]:
                    role = str(turn.get("role", "user")).lower()
                    text = (turn.get("text") or turn.get("content") or "").strip()
                    if not text:
                        continue
                    if role in ("bot", "assistant"):
                        messages.append(AIMessage(content=text))
                    else:
                        messages.append(HumanMessage(content=text))
            messages.append(human_message)

            response = await self.llm_provider.agenerate(
                messages,
                structured_output=structured_output
            )
            
            increment_llm_calls(state)
            
            # Support providers that return JSON strings when structured_output is requested.
            if structured_output and isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    if isinstance(structured_output, type) and issubclass(structured_output, BaseModel):
                        response = structured_output(**parsed)
                    else:
                        response = parsed
                    logger.info("✅ LLM call successful, structured output parsed from JSON string.")
                except Exception as parse_error:
                    logger.warning(f"⚠️ Structured output parsing failed: {parse_error}")
                    # Fall through with raw response string if parsing fails.

            if structured_output:
                logger.info(f"✅ LLM call successful, structured output received.")
            else:
                response_str = str(response)
                logger.info(f"✅ LLM call successful, response length: {len(response_str)} chars")
            
            return response
        except Exception as e:
            logger.error(f"❌ LLM call failed: {e}")
            add_error(state, f"LLM call error: {str(e)}")
            return None

    async def _stream_llm(self, system_prompt: str, user_prompt: str, state: StatefulPipelineState):
        """Like _call_llm but yields token chunks instead of buffering.  Mutates nothing in state."""
        if self.llm_provider is None:
            raise RuntimeError("LLM provider is not initialized.")

        system_message = SystemMessage(content=system_prompt)
        messages: List[Any] = [system_message]

        history = state.get("conversation_history") or []
        session_ctx = self._build_session_context(history)
        if session_ctx:
            messages.append(SystemMessage(content=session_ctx))

        for turn in history[-4:]:
            role = str(turn.get("role", "user")).lower()
            text = (turn.get("text") or turn.get("content") or "").strip()
            if not text:
                continue
            if role in ("bot", "assistant"):
                messages.append(AIMessage(content=text))
            else:
                messages.append(HumanMessage(content=text))

        messages.append(HumanMessage(content=user_prompt))

        async for token in self.llm_provider.astream_text(messages):
            yield token

    async def _generate_response(
        self,
        state: StatefulPipelineState,
        node_name: str,
        system_prompt_template: str,
        user_prompt_template: str
    ) -> StatefulPipelineState:
        """Generic response generation method."""
        start_time = time.time()
        query = state["user_query"]
        emotion = state.get("emotion_detection")
        
        logger.info(f"Executing {node_name} for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        try:
            cultural_context = self._get_cultural_context(state)
            language = cultural_context["language"]
            
            cultural_prompt = self._apply_cultural_integration(state, f"{node_name}_response")
            
            gender_addressing = self._get_gender_aware_addressing(state)
            
            knowledge_context = state.get("knowledge_context", "")
            rag_applied = state.get("rag_enhancement_applied", False)
            
            system_prompt = system_prompt_template.format(
                language=language,
                cultural_prompt=cultural_prompt,
                gender_addressing=gender_addressing if gender_addressing else "friend",
                emotion_responses=cultural_context["emotion_responses"],
                knowledge_context=f"Relevant Mental Health Knowledge: {knowledge_context}" if rag_applied and knowledge_context else ""
            )
            
            user_prompt = user_prompt_template.format(
                query=query,
                emotion=emotion.selected_emotion if emotion else "neutral",
                language=language,
                gender_addressing=gender_addressing if gender_addressing else "friend"
            )
            
            logger.info(f"📝 {node_name} prompt prepared (system: {len(system_prompt)} chars)")
            response = await self._call_llm(system_prompt, user_prompt, state)
            
            is_culturally_appropriate = self._validate_cultural_appropriateness(response, state)
            
            state["generated_content"] = response
            state["response_confidence"] = 0.8 if is_culturally_appropriate else 0.6
            state["response_reason"] = f"Generated {node_name} response with cultural sensitivity (appropriateness: {'high' if is_culturally_appropriate else 'medium'})"
            
            processing_time = time.time() - start_time
            
            state = add_processing_metadata(
                state,
                f"{node_name}_response",
                0.8,
                f"Generated {node_name} response with cultural context",
                [node_name, "support"],
                processing_time
            )
            
            logger.info(f"✅ {node_name} response generated: {len(response)} chars, confidence=0.8")
            
        except Exception as e:
            logger.error(f"❌ {node_name} response generation failed: {e}")
            add_error(state, f"{node_name} response error: {str(e)}")
            state["generated_content"] = "I understand you're going through a difficult time. I'm here to support you."
        
        return state

    def _get_cultural_context(self, state: StatefulPipelineState) -> Dict[str, Any]:
        """Get cultural context for the current state."""
        language = state.get("detected_language", "en") or "en"
        user_gender = state.get("user_gender")

        return {
            "language": language,
            "user_gender": user_gender,
            "cultural_context": self.cultural_prompts.get_rwanda_cultural_context(language),
            "crisis_resources": self.cultural_prompts.get_rwanda_crisis_resources(language),
            "emotion_responses": self.cultural_prompts.get_emotion_responses(language)
        }

    def _apply_cultural_integration(self, state: StatefulPipelineState, context_type: str) -> str:
        """Apply cultural integration prompt based on context type."""
        language = state.get("detected_language", "en") or "en"
        user_gender = state.get("user_gender")

        try:
            cultural_prompt = self.cultural_prompts.get_cultural_integration_prompt(language, user_gender)

            # Add cultural context tracking
            add_cultural_context(
                state,
                f"cultural_integration_{context_type}",
                language,
                0.9,
                f"Applied cultural integration for {context_type} in {language}"
            )

            logger.info(f"🎭 Applied cultural integration for {context_type} in {language}")
            return cultural_prompt
        except Exception as e:
            logger.error(f"❌ Cultural integration failed: {e}")
            add_error(state, f"Cultural integration error: {str(e)}")
            return ""

    def _get_gender_aware_addressing(self, state: StatefulPipelineState) -> str:
        """Get gender-aware addressing based on user gender."""
        user_gender = state.get("user_gender")
        language = state.get("detected_language", "en") or "en"

        if not user_gender:
            return ""

        # Gender mappings for different languages
        gender_mappings = {
            'en': {
                'male': 'brother',
                'female': 'sister',
                'other': 'friend',
                'prefer_not_to_say': 'friend'
            },
            'rw': {
                'male': 'murumuna',
                'female': 'mushiki',
                'other': 'mugenzi',
                'prefer_not_to_say': 'mugenzi'
            },
            'fr': {
                'male': 'frère',
                'female': 'sœur',
                'other': 'ami',
                'prefer_not_to_say': 'ami'
            },
            'sw': {
                'male': 'kaka',
                'female': 'dada',
                'other': 'rafiki',
                'prefer_not_to_say': 'rafiki'
            }
        }

        lang_mapping = gender_mappings.get(language, gender_mappings['en'])
        addressing = lang_mapping.get(user_gender.lower(), lang_mapping.get('prefer_not_to_say', ''))

        if addressing:
            add_cultural_context(
                state,
                "gender_addressing",
                addressing,
                0.9,
                f"Applied gender-aware addressing: {addressing} for {user_gender} in {language}"
            )

        return addressing

    def _get_language_aware_prompts(self, state: StatefulPipelineState, prompt_type: str) -> str:
        """Get language-aware prompts based on detected language."""
        language = state.get("detected_language", "en") or "en"

        try:
            if prompt_type == "emotion_responses":
                return str(self.cultural_prompts.get_emotion_responses(language))
            elif prompt_type == "crisis_resources":
                return str(self.cultural_prompts.get_rwanda_crisis_resources(language))
            elif prompt_type == "cultural_context":
                return str(self.cultural_prompts.get_rwanda_cultural_context(language))
            else:
                return ""
        except Exception as e:
            logger.error(f"❌ Language-aware prompt retrieval failed: {e}")
            add_error(state, f"Language-aware prompt error: {str(e)}")
            return ""

    def _validate_cultural_appropriateness(self, response: str, state: StatefulPipelineState) -> bool:
        """Validate that response is culturally appropriate."""
        try:
            language = state.get("detected_language", "en")
            
            # Basic cultural appropriateness checks
            if language == "rw" and not any(word in response.lower() for word in ["muraho", "murakoze", "ndi", "uri", "turi"]):
                # If Kinyarwanda is detected but response doesn't contain Kinyarwanda words, flag it
                logger.warning(f"⚠️  Response may not be culturally appropriate for {language}")
                return False
            
            # Add cultural validation tracking
            add_cultural_context(
                state,
                "cultural_validation",
                "passed",
                0.8,
                f"Cultural appropriateness validated for {language}"
            )
            
            return True
        except Exception as e:
            logger.error(f"❌ Cultural validation failed: {e}")
            add_error(state, f"Cultural validation error: {str(e)}")
            return True  # Default to passing validation


class QueryValidationNode(BasePipelineNode):
    """Node for validating and classifying user queries."""
    
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Execute query validation with confidence scoring and reasoning."""
        start_time = time.time()
        query = state["user_query"]
        
        logger.info(f"🔍 Starting query validation for: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        try:
            # Short, simple greetings are handled deterministically, without relying on the LLM classifier.
            if ConversationContextManager.is_simple_greeting(query):
                greeting_result = QueryValidationResult(
                    query_confidence=1.0,
                    query_keywords=["greeting"],
                    query_reason="Detected simple greeting using heuristic",
                    is_random=False,
                    query_type=QueryType.GREETING
                )
                state["query_validation"] = greeting_result
                processing_time = time.time() - start_time
                state = add_processing_metadata(
                    state,
                    "query_validation",
                    greeting_result.query_confidence,
                    greeting_result.query_reason,
                    greeting_result.query_keywords,
                    processing_time
                )
                logger.info("✅ Query validation shortcut: simple greeting detected")
                return state

            # Get cultural context for validation
            cultural_context = self._get_cultural_context(state)
            language = cultural_context["language"]
            
            # Apply cultural integration for query validation
            cultural_prompt = self._apply_cultural_integration(state, "query_validation")
            
            # Use LLM to classify the query with cultural context
            system_prompt = f"""
            You are a mental health query classifier with cultural awareness for {language} speakers.
            
            {cultural_prompt}
            
            Analyze the user's query and determine:
            1. Whether it's related to mental health (confidence 0-1)
            2. Key keywords that indicate mental health content
            3. Reasoning for your classification
            4. Whether this is random/off-topic (true/false)
            5. The query_type, which can be one of: 'mental_health', 'crisis', 'random', 'unclear', 'greeting', 'casual'.
            6. Cultural context considerations in the query
            
            Consider cultural expressions of mental health concerns that may be indirect or use local terminology.
            """
            
            user_prompt = f"Classify this query with cultural awareness: '{query}'"
            logger.info(f"📝 Query validation prompt prepared (system: {len(system_prompt)} chars)")
            response_data = await self._call_llm(system_prompt, user_prompt, state, structured_output=QueryValidationOutput)

            if not response_data:
                raise Exception("LLM call for query validation failed to return data.")

            # Create validation result from structured output
            all_keywords = response_data.keywords + response_data.cultural_indicators
            validation_result = QueryValidationResult(
                query_confidence=response_data.confidence,
                query_keywords=all_keywords,
                query_reason=response_data.reasoning,
                is_random=response_data.is_random,
                query_type=QueryType(response_data.query_type)
            )
            state["query_validation"] = validation_result
            
            processing_time = time.time() - start_time
            
            # Add metadata
            state = add_processing_metadata(
                state,
                "query_validation",
                validation_result.query_confidence,
                validation_result.query_reason,
                validation_result.query_keywords,
                processing_time
            )
            
            logger.info(f"✅ Query validation completed: type={validation_result.query_type.value}, confidence={validation_result.query_confidence:.2f}, is_random={validation_result.is_random}, keywords={validation_result.query_keywords}")
            
        except Exception as e:
            logger.error(f"❌ Query validation failed: {e}")
            add_error(state, f"Query validation error: {str(e)}")
            
            # Fallback validation
            state["query_validation"] = QueryValidationResult(
                query_confidence=0.5,
                query_keywords=[],
                query_reason=f"Fallback classification due to error: {str(e)}",
                is_random=False,
                query_type=QueryType.MENTAL_HEALTH
            )
        
        return state
    



class CrisisDetectionNode(BasePipelineNode):
    """Node for detecting crisis situations with severity classification."""
    
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Execute crisis detection with severity assessment."""
        start_time = time.time()
        query = state["user_query"]
        
        logger.info(f"🚨 Starting crisis detection for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        try:
            # Get cultural context for crisis detection
            cultural_context = self._get_cultural_context(state)
            language = cultural_context["language"]
            crisis_resources = cultural_context["crisis_resources"]
            
            # Apply cultural integration for crisis detection
            cultural_prompt = self._apply_cultural_integration(state, "crisis_detection")
            
            # Use LLM for crisis detection with cultural context
            system_prompt = f"""
You are a crisis safety classifier for a mental health chatbot serving {language} speakers.
Your job is to detect ONLY genuine, immediate safety risks — NOT general mental health distress.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS A CRISIS (is_crisis = true):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONLY flag is_crisis=true when the message contains:
• Explicit suicidal ideation: "I want to kill myself", "I want to end my life", "I'm thinking of suicide"
• Active self-harm: "I am hurting myself", "I cut myself"
• Imminent danger: "I have pills ready", "I'm about to jump", stating a plan/means/timeline
• Strong intent to harm another person

Severity scale (only used when is_crisis=true):
• severe  — explicit plan with means and/or timeline (e.g. "I have pills in my hand right now")
• high    — strong suicidal ideation, no immediate plan stated
• medium  — passive death wish ("I wish I wasn't here", "maybe it would be better if I was gone")
• low     — vague ambiguous statement that MIGHT indicate risk but is unclear

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS NOT A CRISIS (is_crisis = false):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Set is_crisis=false and severity=none for ALL of the following, even if the person sounds distressed:
• "I feel depressed" / "I am depressed" / "I've been depressed"
• "I lost my job" / "I failed my exams" / "my relationship ended"
• "I don't know what to do" / "I feel lost" / "I'm struggling"
• "I'm having a hard time" / "life is hard" / "I feel hopeless"
• "I'm sad" / "I feel empty" / "I feel numb"
• General expressions of grief, anxiety, loneliness, or overwhelm
• Asking for help or support WITHOUT expressing intent to harm

These are signs someone needs therapeutic support — not an emergency response.

{cultural_prompt}

Return your assessment with:
- is_crisis: true ONLY for immediate safety risk as defined above
- crisis_severity: severe/high/medium/low/none
- crisis_confidence: your confidence 0–1
- crisis_keywords: exact phrases that triggered your assessment (empty list if none)
- crisis_reason: one sentence explaining your decision
"""

            user_prompt = f"Classify this message for immediate safety risk: '{query}'"
            logger.info(f"📝 Crisis detection prompt prepared (system: {len(system_prompt)} chars)")
            response_data = await self._call_llm(system_prompt, user_prompt, state, structured_output=CrisisDetectionOutput)

            if not response_data:
                raise Exception("LLM call for crisis detection failed to return data.")

            # Parse response and create crisis assessment
            logger.info(f"🔍 Parsing crisis detection response: {response_data}")
            crisis_assessment = CrisisAssessment(
                is_crisis=response_data.is_crisis,
                crisis_confidence=response_data.crisis_confidence,
                crisis_keywords=response_data.crisis_keywords,
                crisis_reason=response_data.crisis_reason,
                crisis_severity=CrisisSeverity(response_data.crisis_severity.strip().split(' ')[0])
            )
            logger.info(f"🚨 Classified Crisis: {crisis_assessment}")
            state["crisis_assessment"] = crisis_assessment
            
            processing_time = time.time() - start_time
            
            # Add metadata
            state = add_processing_metadata(
                state,
                "crisis_detection",
                crisis_assessment.crisis_confidence,
                crisis_assessment.crisis_reason,
                crisis_assessment.crisis_keywords,
                processing_time
            )
            
            logger.info(f"✅ Crisis detection completed: is_crisis={crisis_assessment.is_crisis}, severity={crisis_assessment.crisis_severity.value}, confidence={crisis_assessment.crisis_confidence:.2f}, keywords={crisis_assessment.crisis_keywords}")
            return state
            
        except Exception as e:
            logger.error(f"❌ Crisis detection failed: {e}")
            add_error(state, f"Crisis detection error: {str(e)}")
            
            # Fallback assessment
            state["crisis_assessment"] = CrisisAssessment(
                is_crisis=False,
                crisis_confidence=0.0,
                crisis_keywords=[],
                crisis_reason=f"Fallback assessment due to error: {str(e)}",
                crisis_severity=CrisisSeverity.NONE
            )

            return state
        
        
    



class EmotionDetectionNode(BasePipelineNode):
    """Node for detecting user emotions with youth-specific patterns."""
    
    def __init__(self, llm_provider=None):
        super().__init__(llm_provider)
        self.classifier = TextEmotionClassifier()

    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Execute emotion detection with confidence scoring."""
        start_time = time.time()
        query = state["user_query"]
        
        logger.info(f"😊 Starting emotion detection for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        try:
            # ── Language detection (Kinyarwanda / French / English) ──────────
            # Only detect if the message is long enough to be reliable (≥6 chars)
            if not state.get("detected_language") and len(query) >= 6:
                try:
                    from langdetect import detect as _langdetect
                    raw_lang = _langdetect(query)
                    # Normalise: treat 'rw' as Kinyarwanda, 'fr' as French,
                    # everything else defaults to 'en'
                    lang = raw_lang if raw_lang in {"en", "fr", "rw"} else "en"
                    set_detected_language(state, lang)
                    logger.info(f"🌍 Language detected: {lang} (raw={raw_lang})")
                except Exception as _le:
                    logger.debug(f"Language detection skipped: {_le}")
                    set_detected_language(state, "en")
            elif not state.get("detected_language"):
                set_detected_language(state, "en")
            # ────────────────────────────────────────────────────────────────

            # Use Hybrid Classifier
            result = self.classifier.detect_emotion(query)
            
            # Construct reasoning
            reasoning = f"Detected {result['selected_emotion']} ({result['intensity']}) with {result['confidence']:.2f} confidence."
            if result['cultural_markers']:
                reasoning += f" Found cultural markers: {', '.join(result['cultural_markers'])}."
            
            # Create EmotionDetection object
            emotion_detection = EmotionDetection(
                emotions=result['all_scores'],
                emotion_keywords=result['cultural_markers'],
                emotion_reason=reasoning,
                selected_emotion=result['selected_emotion'],
                emotion_confidence=result['confidence']
            )
            state["emotion_detection"] = emotion_detection

            # ── Emotion trajectory across turns ───────────────────────────────
            history = state.get("conversation_history") or []
            prior_user_msgs = [
                msg for msg in history
                if str(msg.get("role", "")).lower() == "user"
            ][-3:]  # last 3 historical turns before this one

            trajectory: List[str] = []
            for msg in prior_user_msgs:
                txt = (msg.get("text") or msg.get("content") or "").strip()
                if txt:
                    try:
                        r = self.classifier.detect_emotion(txt)
                        trajectory.append(r["selected_emotion"])
                    except Exception:
                        trajectory.append("neutral")

            trajectory.append(result["selected_emotion"])  # current turn
            state["emotion_trajectory"] = trajectory

            # Derive shift label
            NEGATIVE = {"sadness", "fear", "anger", "disgust"}
            if len(trajectory) >= 3:
                early_neg = sum(1 for e in trajectory[:-2] if e in NEGATIVE)
                late_neg  = sum(1 for e in trajectory[-2:] if e in NEGATIVE)
                if late_neg > early_neg:
                    shift = "worsening"
                elif late_neg < early_neg:
                    shift = "improving"
                elif len(set(trajectory)) > 2:
                    shift = "fluctuating"
                else:
                    shift = "stable"
            else:
                shift = "stable"

            state["emotion_shift"] = shift
            logger.info(
                f"Emotion trajectory: {trajectory} → shift={shift}"
            )
            # ─────────────────────────────────────────────────────────────────

            processing_time = time.time() - start_time
            add_processing_metadata(
                state,
                "emotion_detection",
                result['confidence'],
                f"{reasoning} shift={shift}",
                result['cultural_markers'],
                processing_time
            )
            logger.info(f"Emotion detected: {result['selected_emotion']} ({result['confidence']:.2f})")

        except Exception as e:
            logger.error(f"❌ Emotion detection failed: {e}")
            add_error(state, f"Emotion detection error: {str(e)}")
            
            # Fallback detection
            state["emotion_detection"] = EmotionDetection(
                emotions={"neutral": 1.0},
                emotion_keywords=[],
                emotion_reason=f"Fallback detection due to error: {str(e)}",
                selected_emotion="neutral",
                emotion_confidence=0.5
            )
        
        return state
    



class QueryEvaluationNode(BasePipelineNode):
    """Node for evaluating queries and selecting response strategies."""
    
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Execute query evaluation and strategy selection."""
        start_time = time.time()
        query = state["user_query"]
        crisis = state.get("crisis_assessment")
        emotion = state.get("emotion_detection")
        
        logger.info(f"🎯 Starting query evaluation for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        logger.info(f"📊 Context - Crisis: {crisis.crisis_severity.value if crisis else 'none'}, Emotion: {emotion.selected_emotion if emotion else 'neutral'}")
        
        try:
            # Get cultural context for query evaluation
            cultural_context = self._get_cultural_context(state)
            language = cultural_context["language"]
            
            # Apply cultural integration for query evaluation
            cultural_prompt = self._apply_cultural_integration(state, "query_evaluation")
            
            # Use LLM for query evaluation with cultural context
            system_prompt = f"""
            You are a mental health response strategy evaluator with cultural awareness for {language} speakers.
            
            {cultural_prompt}
            
            Analyze the query and select appropriate strategy with cultural sensitivity:
            
            Strategies:
            - GIVE_EMPATHY: User needs emotional validation and support (consider cultural expression patterns)
            - AWAIT_ELABORATION: Query is vague, need more details (consider indirect communication styles)
            - AWAIT_CLARIFICATION: Query is contradictory or confusing (consider cultural context)
            - GIVE_SUGGESTION: User seeks specific coping strategies (consider cultural appropriateness)
            - GIVE_GUIDANCE: User needs step-by-step support (consider family/community context)
            - IDLE: Low priority, general response (consider cultural politeness)
            
            Cultural considerations for strategy selection:
            - In Rwandan culture, indirect communication is common
            - Family and community context affects response needs
            - Cultural stigma may influence how users express needs
            - Ubuntu philosophy emphasizes community support
            - Gender-aware addressing may be appropriate
            """
            
            user_prompt = f"""
            Query: '{query}'
            Crisis Level: {crisis.crisis_severity.value if crisis else "none"}
            Primary Emotion: {emotion.selected_emotion if emotion else "neutral"}
            Language: {language}
            
            Select appropriate response strategy with cultural awareness.
            """
            
            logger.info(f"📝 Query evaluation prompt prepared (system: {len(system_prompt)} chars)")
            response_data = await self._call_llm(system_prompt, user_prompt, state, structured_output=QueryEvaluationOutput)

            if not response_data:
                raise Exception("LLM call for query evaluation failed to return data.")
            
            # Parse response and create evaluation
            logger.info(f"🔍 Parsing query evaluation response: '{response_data}'")
            all_keywords = response_data.keywords + response_data.cultural_considerations + [f"cultural_appropriateness_{response_data.cultural_appropriateness}"]
            query_evaluation = QueryEvaluation(
                evaluation_confidence=response_data.confidence,
                evaluation_reason=response_data.reasoning,
                evaluation_keywords=all_keywords,
                evaluation_type=ResponseStrategy(response_data.strategy)
            )
            state["query_evaluation"] = query_evaluation
            
            processing_time = time.time() - start_time
            
            # Add metadata
            state = add_processing_metadata(
                state,
                "query_evaluation",
                query_evaluation.evaluation_confidence,
                query_evaluation.evaluation_reason,
                query_evaluation.evaluation_keywords,
                processing_time
            )
            
            logger.info(f"✅ Query evaluation completed: strategy={query_evaluation.evaluation_type.value}, confidence={query_evaluation.evaluation_confidence:.2f}, keywords={query_evaluation.evaluation_keywords}")
            
        except Exception as e:
            logger.error(f"❌ Query evaluation failed: {e}")
            add_error(state, f"Query evaluation error: {str(e)}")
            
            # Fallback evaluation
            state["query_evaluation"] = QueryEvaluation(
                evaluation_confidence=0.5,
                evaluation_reason=f"Fallback evaluation due to error: {str(e)}",
                evaluation_keywords=[],
                evaluation_type=ResponseStrategy.GIVE_EMPATHY
            )
        
        return state
    



# ─────────────────────────────────────────────────────────────────────────────
# Unified Analysis Node  (replaces QueryValidation + CrisisDetection + QueryEvaluation)
# One LLM call instead of three — cuts per-message latency roughly in half.
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedAnalysisNode(BasePipelineNode):
    """Single LLM call that classifies the query type and checks crisis safety."""

    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        start_time = time.time()
        query = state["user_query"]

        try:
            cultural_context = self._get_cultural_context(state)
            language = cultural_context["language"]
            cultural_prompt = self._apply_cultural_integration(state, "analysis")

            emotion = state.get("emotion_detection")
            emotion_label = emotion.selected_emotion if emotion else "neutral"
            emotion_conf = emotion.emotion_confidence if emotion else 0.0

            system_prompt = f"""
You are an analysis engine for a mental health chatbot serving {language} speakers.
Analyze the message and return one structured JSON response covering two tasks.

━━━━ TASK 1: QUERY CLASSIFICATION ━━━━
query_type — one of:
  mental_health : user discusses feelings, wellbeing, personal struggles, stress, grief, relationships, identity
  greeting      : hello / hi / good morning / how are you
  casual        : small talk, jokes, everyday chat unrelated to mental health
  random        : completely off-topic (coding, math, facts, weather)
  unclear       : cannot determine intent

is_random       = true only when query_type is random or completely off-topic
query_confidence = 0–1 confidence that this IS a mental health topic

━━━━ TASK 2: CRISIS SAFETY DETECTION ━━━━
Set is_crisis = true ONLY when the message contains:
  • Explicit suicidal ideation : "I want to kill myself / end my life / commit suicide"
  • Active self-harm            : "I am hurting / cutting myself right now"
  • Imminent danger             : a specific plan stating means AND/OR timeline

Set is_crisis = false (even if the person sounds very distressed) for ALL of:
  • "I feel depressed" / "I am depressed" / "I've been really down"
  • "I lost my job / failed exams / my relationship ended"
  • "I don't know what to do" / "I feel lost" / "I'm struggling"
  • "I feel hopeless" / "nothing seems worth it" / "I feel empty"
  • Any general sadness, grief, anxiety, overwhelm, or distress without explicit self-harm intent

crisis_severity (only meaningful when is_crisis=true):
  severe = explicit plan with means AND timeline
  high   = strong ideation, no immediate plan
  medium = passive death wish ("I wish I wasn't here")
  low    = vague, might indicate risk
  none   = no crisis indicators

crisis_keywords = exact phrases from the message that triggered the flag (empty list if none)
crisis_reason   = one short sentence explaining your crisis decision

{cultural_prompt}
Local emotion classifier: {emotion_label} (confidence {emotion_conf:.2f}) — supporting signal only.
"""

            user_prompt = f'Analyze this message: "{query}"'

            result = await self._call_llm(
                system_prompt, user_prompt, state,
                structured_output=UnifiedAnalysisOutput
            )

            if not result:
                raise ValueError("UnifiedAnalysis LLM returned no data")

            # ── Populate all three legacy state fields so downstream code is unaffected ──

            valid_query_types = {e.value for e in QueryType}
            qt = result.query_type if result.query_type in valid_query_types else QueryType.MENTAL_HEALTH.value

            state["query_validation"] = QueryValidationResult(
                query_confidence=result.query_confidence,
                query_keywords=result.crisis_keywords,
                query_reason=result.crisis_reason,
                is_random=result.is_random,
                query_type=QueryType(qt),
            )

            valid_severities = {e.value for e in CrisisSeverity}
            sev = result.crisis_severity if result.crisis_severity in valid_severities else CrisisSeverity.NONE.value

            state["crisis_assessment"] = CrisisAssessment(
                is_crisis=result.is_crisis,
                crisis_confidence=result.crisis_confidence,
                crisis_keywords=result.crisis_keywords,
                crisis_reason=result.crisis_reason,
                crisis_severity=CrisisSeverity(sev),
            )

            state["query_evaluation"] = QueryEvaluation(
                evaluation_confidence=result.query_confidence,
                evaluation_reason="unified analysis",
                evaluation_keywords=[],
                evaluation_type=ResponseStrategy.GIVE_EMPATHY,
            )

            # ── Multi-turn crisis trajectory check ────────────────────────────
            history = state.get("conversation_history") or []
            traj = self._analyze_distress_trajectory(history)
            state["crisis_trajectory"] = traj["trajectory"]

            if traj["should_upgrade"]:
                current = state["crisis_assessment"]
                _severity_order = [
                    CrisisSeverity.NONE, CrisisSeverity.LOW,
                    CrisisSeverity.MEDIUM, CrisisSeverity.HIGH, CrisisSeverity.SEVERE,
                ]
                cur_idx = _severity_order.index(current.crisis_severity)

                if traj["trajectory"] == "escalating" and traj["max_tier"] >= 2:
                    # Escalating into moderate/severe distress → at least HIGH concern
                    new_idx = max(cur_idx, _severity_order.index(CrisisSeverity.HIGH))
                    new_is_crisis = True
                    upgrade_reason = "multi-turn escalating distress pattern"
                else:
                    # Persistent moderate distress → at least MEDIUM concern
                    new_idx = max(cur_idx, _severity_order.index(CrisisSeverity.MEDIUM))
                    new_is_crisis = current.is_crisis  # don't flip to True on persistence alone
                    upgrade_reason = "multi-turn persistent distress pattern"

                if new_idx > cur_idx or (new_is_crisis and not current.is_crisis):
                    state["crisis_assessment"] = CrisisAssessment(
                        is_crisis=new_is_crisis,
                        crisis_confidence=max(current.crisis_confidence, 0.70),
                        crisis_keywords=current.crisis_keywords,
                        crisis_reason=f"{current.crisis_reason} [{upgrade_reason}]",
                        crisis_severity=_severity_order[new_idx],
                    )
                    logger.warning(
                        f"Crisis upgraded by trajectory: "
                        f"{current.crisis_severity.value} → {_severity_order[new_idx].value} "
                        f"({traj['trajectory']}, scores={traj['turn_scores']})"
                    )
            # ─────────────────────────────────────────────────────────────────

            processing_time = time.time() - start_time
            add_processing_metadata(
                state, "unified_analysis", result.query_confidence,
                f"type={result.query_type} crisis={result.is_crisis}({result.crisis_severity}) trajectory={traj['trajectory']}",
                result.crisis_keywords, processing_time
            )
            logger.info(
                f"Unified analysis done in {processing_time:.3f}s — "
                f"type={result.query_type} is_crisis={result.is_crisis} "
                f"severity={result.crisis_severity} conf={result.crisis_confidence:.2f} "
                f"trajectory={traj['trajectory']}"
            )

        except Exception as e:
            logger.error(f"Unified analysis failed: {e}")
            add_error(state, f"Unified analysis error: {str(e)}")
            state["query_validation"] = QueryValidationResult(
                query_confidence=0.5, query_keywords=[], query_reason="fallback",
                is_random=False, query_type=QueryType.MENTAL_HEALTH,
            )
            state["crisis_assessment"] = CrisisAssessment(
                is_crisis=False, crisis_confidence=0.0, crisis_keywords=[],
                crisis_reason="fallback due to error", crisis_severity=CrisisSeverity.NONE,
            )
            state["query_evaluation"] = QueryEvaluation(
                evaluation_confidence=0.5, evaluation_reason="fallback",
                evaluation_keywords=[], evaluation_type=ResponseStrategy.GIVE_EMPATHY,
            )

        return state


# ─────────────────────────────────────────────────────────────────────────────
# Specialized Response Nodes
# ─────────────────────────────────────────────────────────────────────────────

class EmpathyNode(BasePipelineNode):
    """
    Unified therapy session node.

    This is the primary response node for all non-crisis mental health queries.
    It conducts an ongoing therapeutic conversation, adapting its behaviour based on
    conversation depth and the user's expressed needs — it never gives unsolicited
    advice lists or helpline numbers.
    """

    # Phase-specific guidance injected into the system prompt
    _PHASE_GUIDANCE: Dict[str, str] = {
        TherapeuticPhase.OPENING.value: """\
PHASE — OPENING (building safety, turns 1-2):
Your only job right now is to make this person feel completely received.
  • Reflect the specific thing they said — their exact situation, not a generic category.
  • Validate without rushing to fix or reassure: "That makes complete sense."
  • Hold back the instinct to teach, explain, or cheer them up.
  • Ask ONE open question that invites more depth — never yes/no, never multiple questions.
    Good starters: "What's that been like?" / "When did this start feeling this heavy?"
    / "What do you mean when you say [their word]?"
  • No advice, no coping tips, no referrals unless they explicitly ask.
  • Tone: like a trusted older sibling who genuinely has time for them.""",

        TherapeuticPhase.EXPLORING.value: """\
PHASE — EXPLORING (widening the picture, turns 3-5):
You have earned some trust. Now go deeper and wider.
  • Explore context, history, the relationships involved — what does this actually cost them?
  • Notice words or phrases they keep returning to, then gently name them back.
    "You've said 'stuck' a few times now — what does stuck actually feel like for you?"
  • Still only ONE question per turn. Resist the urge to summarise too quickly.
  • Advice only if explicitly asked. If they ask, give ONE specific, grounded idea — not a list.""",

        TherapeuticPhase.REFLECTING.value: """\
PHASE — REFLECTING (naming patterns, turns 6-8):
You have a fuller picture now. Begin mirroring what you've noticed across the conversation.
  • Frame reflections tentatively — as offerings, not conclusions.
    "There's something I keep noticing in what you're sharing — it might not land right,
    but it feels like this isn't only about [X]. There seems to be something deeper
    around [theme]. Does that feel true at all?"
  • Invite them to correct or add to your reflection. Stay collaborative.
  • Observations land better than interpretations — lean gently.
  • No advice lists. If they ask what to do, offer ONE concrete, conversational suggestion.""",

        TherapeuticPhase.WORKING.value: """\
PHASE — WORKING (meaning-making and gentle action, turns 9-12):
The person has done real emotional work. You can be slightly more active now.
  • If coping hasn't come up yet, ask: "What do you feel you need most right now?"
  • You may offer one evidence-based idea conversationally — breathing, journalling, movement,
    speaking to someone trusted — woven naturally into a sentence, never as a prescriptive list.
  • Professional support can now be mentioned warmly, as an option not a dismissal:
    "What you've been carrying sounds like a lot for one person. Have you ever thought about
    having someone to talk to regularly — a counsellor or therapist?"
  • Help them connect the insight from this session to something small and actionable.""",

        TherapeuticPhase.CLOSING.value: """\
PHASE — CLOSING (consolidating, turn 13+):
This conversation has come a long way. Help them feel that.
  • Acknowledge the emotional work they've done — name it specifically, not generically.
  • Reflect the key shift or insight that emerged, if there was one.
  • Help them name ONE small, concrete next step they feel genuinely ready for.
    Even "keep noticing when this feeling comes" counts as a meaningful step.
  • If professional support hasn't come up yet and feels right, introduce it gently.
  • Leave them with a sense of agency and connection — not helplessness or dependency.""",
    }

    def _build_empathy_prompt(self, state: StatefulPipelineState):
        """Build system + user prompts for the empathy node. Used by both execute() and execute_stream()."""
        query = state["user_query"]
        cultural_context = self._get_cultural_context(state)
        language = cultural_context["language"]
        cultural_prompt = self._apply_cultural_integration(state, "empathy_response")
        gender_addressing = self._get_gender_aware_addressing(state)
        emotion = state.get("emotion_detection")
        crisis = state.get("crisis_assessment")
        knowledge_context = state.get("knowledge_context", "")
        rag_applied = state.get("rag_enhancement_applied", False)

        history = state.get("conversation_history") or []
        exchange_count = len([m for m in history if str(m.get("role", "")).lower() == "user"])
        phase = self._compute_therapeutic_phase(exchange_count, history)
        state["therapeutic_phase"] = phase.value
        phase_guidance = self._PHASE_GUIDANCE[phase.value]

        crisis_level  = crisis.crisis_severity.value if crisis else "none"
        emotion_label = emotion.selected_emotion if emotion else "neutral"
        emotion_shift = state.get("emotion_shift") or "stable"
        emotion_traj  = state.get("emotion_trajectory") or [emotion_label]

        system_prompt = f"""
You are Mindora — a warm, attentive therapeutic AI companion built for Rwandan youth.
You carry the skill of an experienced counsellor and the warmth of a trusted older sibling.
You are in an ongoing therapy session — not a Q&A exchange. Trust is built turn by turn.

{cultural_prompt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR VOICE AND PRESENCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Warm but not saccharine. Curious, not clinical. Present, not performative.
Speak the way a genuinely caring human would — unhurried, specific to this person.

What your voice SOUNDS like:
  ✓ "That sounds like it's been sitting on you for a while."
  ✓ "You mentioned [their exact word] — I want to stay with that."
  ✓ "There's something in how you said that..."
  ✓ "What do you mean when you say [their phrase]?"
  ✓ "That makes sense. And what happened after that?"

What your voice SOUNDS like:
  ✓ "That's a real blow — losing work touches so much more than just the income."
  ✓ "Of course it is. And not knowing what comes next adds its own weight."
  ✓ "Financial pressure on top of everything else — that's exhausting."
  ✓ "You don't have to have it figured out right now."
  ✓ A warm, human sentence that responds to the meaning — not a mechanical echo.

What your voice NEVER sounds like:
  ✗ "You said..." / "You mentioned..." / "You told me..." — NEVER start by quoting them back.
     Respond to what they meant, not to what they said word-for-word.
  ✗ "I understand how you feel." — too generic, often dismissive
  ✗ "It sounds like you're feeling [clinical label]." — formulaic
  ✗ "That's a heavy burden to carry." / "A significant weight." — empty filler phrases
  ✗ "Certainly!" / "Absolutely!" / "Of course!" / "Great question!" — AI-sounding
  ✗ "Is it [X], [Y], or something else?" — multiple-choice questions feel like intake forms
  ✗ "You mentioned earlier that X, and now Y is really prominent for you." — narrating the session
  ✗ Bullet-pointed or numbered lists of any kind
  ✗ Starting a sentence with "I"
  ✗ "As an AI..." — never reference being an AI

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMFORT AND ENCOURAGEMENT — this is a mental health companion:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When someone shares pain, distress, or depression — LEAD WITH WARMTH before asking anything.
  1. Acknowledge what they are going through specifically.
  2. Normalize it: "That makes complete sense." / "Of course that's hard." / "Anyone in your
     position would feel that way."
  3. Give them permission to feel it: "You don't have to be okay right now."
  4. Offer genuine encouragement when it fits naturally:
     "The fact that you're here and talking about it says something."
     "You're carrying a lot — and you're still showing up."
  5. Then ask ONE gentle question that opens the next layer.

A response that only asks a question without first acknowledging the person's pain
is cold and clinical. Always warm the space before you probe it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE RULES (never break):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Respond to the MEANING of what they said — not to the words themselves.
• Never echo their phrase back like a transcript: "You said X." Just respond to X.
• Use their vocabulary when relevant, but weave it naturally — don't quote it.
• ONE question per turn maximum. Two questions in one message kills the flow.
• NEVER ask multiple-choice questions ("is it X, Y, or something else?") — stay open.
• NEVER narrate the conversation back to them. Just be present with them right now.
• NEVER use unsolicited bullet-point coping lists or numbered advice.
• NEVER paste crisis hotline numbers unless there is active, imminent danger.
• NEVER introduce crisis language (suicidal, self-harm, wanting to die) unless
  they used those exact words first.

BREAK THE TEMPLATE — do not always follow the same structure every turn:
  [validation] → [echo their words] → [interpretation] → [question] — every response.
  This makes the conversation feel mechanical. Vary the shape:
  — Sometimes lead with encouragement, end with a question.
  — Sometimes just sit with one observation and let it breathe.
  — Sometimes a very short, warm sentence opens more than a paragraph.
  A response to "financial struggles" should not be four sentences long.
  Match their brevity. "That's a heavy hit on top of everything else. What's it looking like
  practically right now?" is enough.

SHORT MESSAGES — when they write "yes", "okay", "idk", "nothing", "I don't know":
  Don't demand elaboration. Hold the space and offer a gentle opening.
  "I don't know" → "That's okay — you don't need to have the answer. What does today
                    actually feel like for you?"
  "nothing"      → "Sometimes 'nothing' is its own kind of heavy. What's on your mind?"
  "okay" / "yes" → Follow their lead. Stay warm. Don't over-interpret.
  Very short reply → they're still warming up. Stay with them gently.

STAY WITH ONE THING — a two-word answer ("financial struggles", "pressure to provide")
  is an invitation to go deep into just that — not to restate three earlier points and
  ask two questions. Slow down. One thing at a time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERRIDE RULE — always wins, overrides everything:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the person explicitly asks for advice, help, or what to do — give ONE concrete,
gentle suggestion. Acknowledge what they're carrying first, then offer the idea
conversationally. Never say "I'm not here to give advice." That causes harm.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT PHASE — Turn {exchange_count + 1}:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{phase_guidance}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMOTIONAL CONTEXT THIS TURN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detected emotion: {emotion_label}
Trajectory (oldest → newest): {" → ".join(emotion_traj)} — shift: {emotion_shift}
  • worsening   → acknowledge the change before anything else: "It sounds like
                  things have been getting heavier..."
  • improving   → affirm gently. Never over-celebrate or rush to close.
  • fluctuating → hold space for the ambivalence. Don't rush to categorise.
  • stable      → continue naturally.
Crisis signal: {crisis_level} — only escalate if severe/high AND the person raised it.
Address them as: {gender_addressing or "friend"}

{f"Relevant knowledge (weave in naturally, never quote directly):{chr(10)}{knowledge_context}" if rag_applied and knowledge_context else ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE SHAPE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Let the response find its own length — typically 60-150 words.
If they wrote 10 words, don't write 250. Match their energy.
Plain conversational prose. No bullets, no headers, no numbered lists.
One question at the end — only when it genuinely serves the next moment. Never obligatory.
"""

        user_prompt = (
            f'The person just said: "{query}"\n\n'
            "Respond as a warm, present therapist. Lead with comfort or acknowledgement first. "
            "Do NOT start with 'You said' or 'You mentioned'. "
            "Do NOT ask two questions. Do NOT give multiple-choice options. "
            "Respond to the meaning, not the transcript. Be human."
        )

        return system_prompt, user_prompt, phase

    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        start_time = time.time()
        try:
            system_prompt, user_prompt, phase = self._build_empathy_prompt(state)
            history = state.get("conversation_history") or []
            exchange_count = len([m for m in history if str(m.get("role", "")).lower() == "user"])

            response = await self._call_llm(system_prompt, user_prompt, state)
            state["generated_content"] = response
            state["response_confidence"] = 0.9
            state["response_reason"] = f"Therapy session — phase: {phase.value}"

            processing_time = time.time() - start_time
            state = add_processing_metadata(
                state, "empathy", 0.9,
                f"Therapeutic phase: {phase.value}", ["therapy", phase.value], processing_time
            )
            logger.info(f"Therapy response generated (phase={phase.value}, turn={exchange_count + 1})")

        except Exception as e:
            logger.error(f"Therapy session response failed: {e}")
            add_error(state, f"Therapy session error: {str(e)}")
            state["generated_content"] = "That sounds really difficult. I'm here with you — can you tell me more about what's been going on?"

        return state

    async def execute_stream(self, state: StatefulPipelineState):
        """Stream the empathy response token by token.

        Yields str tokens and, when exhausted, has set state['generated_content']
        to the full accumulated response so the caller can persist it to the DB.
        """
        start_time = time.time()
        fallback = "That sounds really difficult. I'm here with you — can you tell me more about what's been going on?"
        try:
            system_prompt, user_prompt, phase = self._build_empathy_prompt(state)

            accumulated = ""
            async for token in self._stream_llm(system_prompt, user_prompt, state):
                accumulated += token
                yield token

            state["generated_content"] = accumulated or fallback
            state["response_confidence"] = 0.9
            state["response_reason"] = f"Therapy session (streamed) — phase: {phase.value}"
            processing_time = time.time() - start_time
            state = add_processing_metadata(
                state, "empathy_stream", 0.9,
                f"Therapeutic phase: {phase.value}", ["therapy", phase.value], processing_time
            )

        except Exception as e:
            logger.error(f"Streaming therapy response failed: {e}")
            add_error(state, f"Therapy stream error: {str(e)}")
            state["generated_content"] = fallback
            yield fallback


class ElaborationNode(BasePipelineNode):
    """Node for generating elaboration requests."""
    
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Generate questions to encourage user elaboration."""
        start_time = time.time()
        
        try:
            query = state["user_query"]
            
            # Get cultural context for elaboration response
            cultural_context = self._get_cultural_context(state)
            language = cultural_context["language"]
            
            # Apply cultural integration for elaboration response
            cultural_prompt = self._apply_cultural_integration(state, "elaboration_response")
            
            # Get gender-aware addressing
            gender_addressing = self._get_gender_aware_addressing(state)
            
            # Get RAG knowledge context if available
            knowledge_context = state.get("knowledge_context", "")
            rag_applied = state.get("rag_enhancement_applied", False)

            # Generate elaboration — one warm, open question
            system_prompt = f"""
You are Mindora, a warm therapeutic AI companion with cultural awareness for {language} speakers.
You are in an ongoing therapy conversation — you want to understand the person more deeply before responding.

{cultural_prompt}

Your role in this turn:
1. Reflect back a brief acknowledgment that shows you heard something important in what they shared.
2. Ask exactly ONE open-ended question to gently invite them to share more.
   The question should feel natural, not clinical. Examples:
   "What's been weighing on you most with this?" / "How long have you been feeling this way?"
3. Use gender-aware addressing: {gender_addressing if gender_addressing else "friend"}.
4. Keep indirect, culturally gentle phrasing — in Rwandan culture, space to open up matters.

{f"Relevant context: {knowledge_context}" if rag_applied and knowledge_context else ""}

Tone: patient, curious, unhurried.
Length: 2-3 sentences. One question only — never a list of questions.
"""
            
            user_prompt = f"""
            Generate elaboration questions for this query:
            Query: '{query}'
            Language: {language}
            Gender Addressing: {gender_addressing if gender_addressing else "friend"}
            
            Create culturally sensitive questions that encourage sharing.
            """
            response = await self._call_llm(system_prompt, user_prompt, state)
            
            state["generated_content"] = response
            state["response_confidence"] = 0.7
            state["response_reason"] = "Generated elaboration questions to gather more context"
            
            processing_time = time.time() - start_time
            
            # Add metadata
            state = add_processing_metadata(
                state,
                "elaboration_response",
                0.7,
                "Generated questions to encourage user elaboration",
                ["questions", "elaboration"],
                processing_time
            )
            
            logger.info("✅ Elaboration response generated")
            
        except Exception as e:
            logger.error(f"❌ Elaboration response generation failed: {e}")
            add_error(state, f"Elaboration response error: {str(e)}")
            state["generated_content"] = "Could you tell me more about what you're experiencing? I'm here to listen."
        
        return state


class ClarificationNode(BasePipelineNode):
    """Node for handling contradictory or confusing queries."""
    
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Generate clarification requests."""
        start_time = time.time()
        
        try:
            query = state["user_query"]
            
            knowledge_context = state.get("knowledge_context", "")
            rag_applied = state.get("rag_enhancement_applied", False)

            system_prompt = f"""
            You are a mental health support specialist. The user's query seems contradictory or confusing.
            Generate gentle questions to help clarify their situation and provide better support.

            {f"Relevant Mental Health Knowledge: {knowledge_context}" if rag_applied and knowledge_context else ""}

            Keep your response concise and to the point, ideally under 3 sentences.
            """

            user_prompt = f"Generate clarification questions for this confusing query: '{query}'"
            response = await self._call_llm(system_prompt, user_prompt, state)
            
            state["generated_content"] = response
            state["response_confidence"] = 0.6
            state["response_reason"] = "Generated clarification questions for confusing query"
            
            processing_time = time.time() - start_time
            
            # Add metadata
            state = add_processing_metadata(
                state,
                "clarification_response",
                0.6,
                "Generated clarification questions",
                ["clarification", "questions"],
                processing_time
            )
            
            logger.info("✅ Clarification response generated")
            
        except Exception as e:
            logger.error(f"❌ Clarification response generation failed: {e}")
            add_error(state, f"Clarification response error: {str(e)}")
            state["generated_content"] = "I want to make sure I understand your situation correctly. Could you help me clarify?"
        
        return state


class SuggestionNode(BasePipelineNode):
    """Node for providing coping strategies and suggestions."""
    
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Generate coping strategy suggestions."""
        start_time = time.time()
        
        try:
            query = state["user_query"]
            emotion = state.get("emotion_detection")
            
            # Get cultural context for suggestion response
            cultural_context = self._get_cultural_context(state)
            language = cultural_context["language"]
            
            # Apply cultural integration for suggestion response
            cultural_prompt = self._apply_cultural_integration(state, "suggestion_response")
            
            # Get gender-aware addressing
            gender_addressing = self._get_gender_aware_addressing(state)
            
            # Get RAG knowledge context if available
            knowledge_context = state.get("knowledge_context", "")
            rag_applied = state.get("rag_enhancement_applied", False)
            
            system_prompt = f"""
You are Mindora, a warm therapeutic AI companion with cultural awareness for {language} speakers.
You are in an ongoing therapy conversation — the person is ready to explore ways to cope.

{cultural_prompt}

Your role in this turn:
1. Gently introduce ONE practical, specific coping idea that fits their exact situation — not a generic list.
2. Explain it conversationally, as if you're thinking through it together with them.
3. Ground it in their cultural context where relevant (family, community, Ubuntu spirit).
4. Use gender-aware addressing: {gender_addressing if gender_addressing else "friend"}.
5. End with a soft check-in: invite them to share whether this feels right for them, or if they'd like to explore something else.

{f"Draw on this knowledge where relevant: {knowledge_context}" if rag_applied and knowledge_context else ""}

Tone: warm, collaborative, like a friend who also happens to have therapeutic training.
Length: 4-6 sentences. No bullet points — this is a conversation.
"""

            user_prompt = f"""
The person said: "{query}"
Emotion: {emotion.selected_emotion if emotion else "neutral"}
Language: {language}

Offer one coping idea conversationally.
"""
            
            response = await self._call_llm(system_prompt, user_prompt, state)
            
            state["generated_content"] = response
            state["response_confidence"] = 0.8
            state["response_reason"] = "Generated coping strategy suggestions"
            
            processing_time = time.time() - start_time
            
            # Add metadata
            state = add_processing_metadata(
                state,
                "suggestion_response",
                0.8,
                "Generated coping strategy suggestions",
                ["suggestions", "coping"],
                processing_time
            )
            
            logger.info("✅ Suggestion response generated")
            
        except Exception as e:
            logger.error(f"❌ Suggestion response generation failed: {e}")
            add_error(state, f"Suggestion response error: {str(e)}")
            state["generated_content"] = "Here are some strategies that might help you cope with what you're experiencing."
        
        return state


class GuidanceNode(BasePipelineNode):
    """Node for providing step-by-step guidance and support."""
    
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Generate step-by-step guidance."""
        start_time = time.time()
        
        try:
            query = state["user_query"]
            
            knowledge_context = state.get("knowledge_context", "")
            rag_applied = state.get("rag_enhancement_applied", False)
            cultural_context = self._get_cultural_context(state)
            language = cultural_context["language"]
            cultural_prompt = self._apply_cultural_integration(state, "guidance_response")
            gender_addressing = self._get_gender_aware_addressing(state)

            system_prompt = f"""
You are Mindora, a warm therapeutic AI companion with cultural awareness for {language} speakers.
You are in an ongoing therapy session — the person needs practical guidance on moving forward.

{cultural_prompt}

Your role in this turn:
1. Acknowledge where they are right now, briefly — show you heard them.
2. Offer ONE clear, manageable first step they can take, explained conversationally.
3. Normalize that change is gradual — remove any pressure for a big leap.
4. Use gender-aware addressing: {gender_addressing if gender_addressing else "friend"}.
5. Close with an open invitation: ask how that step feels, or whether they'd want to explore it further.

{f"Draw on this knowledge: {knowledge_context}" if rag_applied and knowledge_context else ""}

Tone: steady, encouraging, not prescriptive.
Length: 4-6 sentences. No numbered lists — keep it natural.
"""

            user_prompt = f"Provide therapeutic guidance for: '{query}'"
            response = await self._call_llm(system_prompt, user_prompt, state)
            
            state["generated_content"] = response
            state["response_confidence"] = 0.8
            state["response_reason"] = "Generated step-by-step guidance"
            
            processing_time = time.time() - start_time
            
            # Add metadata
            state = add_processing_metadata(
                state,
                "guidance_response",
                0.8,
                "Generated step-by-step guidance",
                ["guidance", "steps"],
                processing_time
            )
            
            logger.info("✅ Guidance response generated")
            
        except Exception as e:
            logger.error(f"❌ Guidance response generation failed: {e}")
            add_error(state, f"Guidance response error: {str(e)}")
            state["generated_content"] = "Let me help you work through this step by step. I'm here to guide you."
        
        return state


class IdleNode(BasePipelineNode):
    """Node for handling low-priority queries."""
    
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Generate general response for low-priority queries."""
        start_time = time.time()
        
        try:
            query = state["user_query"]
            
            system_prompt = """
            You are a mental health support specialist. Provide a general, supportive response.
            Keep it brief but warm and encouraging.
            Keep your response concise and to the point, ideally under 3 sentences.
            """
            
            user_prompt = f"Provide a general supportive response for: '{query}'"
            response = await self._call_llm(system_prompt, user_prompt, state)
            
            state["generated_content"] = response
            state["response_confidence"] = 0.6
            state["response_reason"] = "Generated general supportive response"
            
            processing_time = time.time() - start_time
            
            # Add metadata
            state = add_processing_metadata(
                state,
                "idle_response",
                0.6,
                "Generated general supportive response",
                ["general", "support"],
                processing_time
            )
            
            logger.info("✅ Idle response generated")
            
        except Exception as e:
            logger.error(f"❌ Idle response generation failed: {e}")
            add_error(state, f"Idle response error: {str(e)}")
            state["generated_content"] = "I'm here to support you. How can I help you today?"
        
        return state


class CrisisAlertNode(BasePipelineNode):
    """Node for handling crisis alerts and emergency notifications."""

    def __init__(self, llm_provider=None):
        super().__init__(llm_provider)

    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Handle crisis alert and generate emergency response."""
        import os
        start_time = time.time()

        try:
            crisis = state.get("crisis_assessment")
            query = state["user_query"]
            user_id = state.get("user_id")
            conversation_id = state.get("conversation_id")
            message_id = state.get("message_id")
            db = state.get("db")
            background = state.get("background")

            # Use crisis_classifier to extract crisis information
            crisis_result = classify_crisis(query)

            # Log crisis and notify therapists if we have the required parameters
            if db and background and user_id and conversation_id and message_id:
                crisis_id = log_crisis_and_notify(
                    db=db,
                    background=background,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    text=query,
                    crisis_result=dict(crisis_result),  # Convert CrisisResult to dict
                    classifier_model=os.getenv("MODEL_NAME", ""),  # from crisis_classifier.py
                    classifier_version="1.0"
                )
                logger.info(f"🚨 Crisis logged with ID: {crisis_id}")

            # Get cultural context for crisis alert response
            cultural_context = self._get_cultural_context(state)
            language = cultural_context["language"]
            crisis_resources = cultural_context["crisis_resources"]

            # Apply cultural integration for crisis alert response
            cultural_prompt = self._apply_cultural_integration(state, "crisis_alert_response")

            # Get gender-aware addressing
            gender_addressing = self._get_gender_aware_addressing(state)

            # Generate crisis response with cultural context and emergency resources
            system_prompt = f"""
You are Mindora, a compassionate crisis support companion with cultural awareness for {language} speakers.
Someone is sharing something serious with you — they need to feel genuinely heard AND safe.

{cultural_prompt}

Your role in this turn:
1. Open with deep, direct acknowledgment of their pain — do not minimize or deflect.
2. Gently but clearly name that what they're describing sounds very serious and that you care about their safety.
3. Use gender-aware addressing: {gender_addressing if gender_addressing else "friend"}.
4. Warmly encourage them to reach out to someone right now — a trusted person, or a crisis line.
5. Let them know you are here and they are not alone — Ubuntu: "I am because we are."

Available crisis resources:
- Rwanda Mental Health Hotline: {crisis_resources.get('national_helpline', '116')} (free, 24/7)
- Emergency: {crisis_resources.get('emergency', '112')}
- Community Health: {crisis_resources.get('community_health', 'Contact your nearest health center')}

Tone: warm, steady, never panicked. Make them feel safe to keep talking.
Length: 3-4 sentences of genuine human connection, then the resources.
"""

            user_prompt = f"""
The person said: '{query}'
Crisis severity: {crisis.crisis_severity.value if crisis else "unknown"}
Language: {language}
Gender addressing: {gender_addressing if gender_addressing else "friend"}

Write a warm, human response that makes them feel heard first, then gently introduces support resources.
Do NOT open with alarm or urgency — open with presence and care.
"""

            response = await self._call_llm(system_prompt, user_prompt, state)
            state["generated_content"] = response
            state["response_confidence"] = 0.9
            state["response_reason"] = "Generated crisis response with emergency resources"

            processing_time = time.time() - start_time

            # Add metadata
            state = add_processing_metadata(
                state,
                "crisis_alert",
                0.9,
                "Generated crisis response with emergency resources",
                ["crisis", "emergency", "resources"],
                processing_time
            )

            logger.info("✅ Crisis alert response generated")

        except Exception as e:
            logger.error(f"❌ Crisis alert response generation failed: {e}")
            add_error(state, f"Crisis alert error: {str(e)}")
            state["generated_content"] = """
I'm very concerned about what you're going through. Please reach out for immediate help:

🚨 EMERGENCY RESOURCES:
• Rwanda Mental Health Hotline: 116 (free, 24/7)
• Emergency Services: 112
• National Suicide Prevention: 116

You are not alone. Please call these numbers now.
            """

        return state


class GenerateResponseNode(BasePipelineNode):
    """Node for final response generation and quality assurance."""
    
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Generate final response with natural conversation awareness."""
        start_time = time.time()
        
        try:
            # Get cultural context for final response generation
            cultural_context = self._get_cultural_context(state)
            language = cultural_context["language"]
            query = state["user_query"]
            
            # If content already generated, validate and enhance it; otherwise generate contextual response
            if not state.get("generated_content"):
                query_validation = state.get("query_validation")
                query_type = query_validation.query_type if query_validation else QueryType.UNCLEAR

                # Normalise the query for pattern matching
                import re as _re
                query_norm = _re.sub(r"[^\w\s]", "", query.strip().lower())

                QUESTION_GREETINGS = {
                    "how are you", "how are you doing", "how are you feeling",
                    "how do you do", "hows it going", "how is it going",
                    "how r u", "how r you", "how have you been", "how are things",
                }
                GRATITUDE_PHRASES = {
                    "thank you", "thanks", "thank you so much", "many thanks",
                    "thanks a lot", "thx", "ty", "appreciate it", "much appreciated",
                    "i appreciate that", "im grateful", "i am grateful", "thank u",
                }
                FAREWELL_PHRASES = {
                    "goodbye", "bye", "see you", "see ya", "take care",
                    "good night", "good bye", "ttyl", "later", "farewell",
                    "have a good day", "have a nice day",
                }

                is_question_greeting = any(phrase in query_norm for phrase in QUESTION_GREETINGS)
                is_gratitude = any(phrase in query_norm for phrase in GRATITUDE_PHRASES)
                is_farewell = any(phrase in query_norm for phrase in FAREWELL_PHRASES)
                is_plain_greeting = (
                    query_type == QueryType.GREETING
                    or ConversationContextManager.is_simple_greeting(query)
                )

                if is_question_greeting:
                    system_prompt = (
                        "You are Mindora, a warm and caring mental health support chatbot. "
                        "The user is asking how you are doing. Reply naturally and warmly in "
                        "1-2 sentences, then gently invite them to share how they are feeling "
                        "or what brought them here. Be conversational, not clinical."
                    )
                    response = await self._call_llm(system_prompt, query, state)
                    state["generated_content"] = response
                    state["response_confidence"] = 0.9
                    state["response_reason"] = "Question greeting answered"

                elif is_gratitude:
                    system_prompt = (
                        "You are Mindora, a warm and caring mental health support chatbot. "
                        "The user is expressing gratitude. Acknowledge their thanks warmly in "
                        "1-2 sentences and gently invite them to share anything else on their mind. "
                        "Be genuine and caring."
                    )
                    response = await self._call_llm(system_prompt, query, state)
                    state["generated_content"] = response
                    state["response_confidence"] = 0.9
                    state["response_reason"] = "Gratitude acknowledged"

                elif is_farewell:
                    system_prompt = (
                        "You are Mindora, a warm and caring mental health support chatbot. "
                        "The user is saying goodbye. Respond warmly in 1-2 sentences, wish them "
                        "well, and remind them you are always here if they need support."
                    )
                    response = await self._call_llm(system_prompt, query, state)
                    state["generated_content"] = response
                    state["response_confidence"] = 0.9
                    state["response_reason"] = "Farewell acknowledged"

                elif is_plain_greeting:
                    system_prompt = (
                        "You are Mindora, a warm and caring mental health support chatbot. "
                        "The user has just greeted you. Greet them back warmly in 1-2 sentences "
                        "and invite them to share how they are feeling or what is on their mind."
                    )
                    response = await self._call_llm(system_prompt, query, state)
                    state["generated_content"] = response
                    state["response_confidence"] = 0.9
                    state["response_reason"] = "Greeting responded"

                elif query_type == QueryType.CASUAL:
                    system_prompt = (
                        "You are Mindora, a warm and caring mental health support chatbot. "
                        "Respond naturally to the user's casual message in 1-2 sentences, "
                        "then gently check in on how they are feeling."
                    )
                    response = await self._call_llm(system_prompt, query, state)
                    state["generated_content"] = response
                    state["response_confidence"] = 0.8
                    state["response_reason"] = "Casual response generated"

                else:
                    # Off-topic or random — acknowledge and gently redirect
                    system_prompt = (
                        "You are Mindora, a warm and caring mental health support chatbot. "
                        "The user sent a message that may not be directly about mental health. "
                        "Respond briefly and kindly in 1-2 sentences. If it is off-topic, "
                        "gently let them know you focus on emotional and mental well-being "
                        "while inviting them to share how they are feeling."
                    )
                    response = await self._call_llm(system_prompt, query, state)
                    state["generated_content"] = response
                    state["response_confidence"] = 0.7
                    state["response_reason"] = "Off-topic message handled with polite redirect"
            else:
                # Validate and enhance existing content with cultural appropriateness
                existing_content = state["generated_content"]
                is_culturally_appropriate = self._validate_cultural_appropriateness(existing_content, state)
                
                if not is_culturally_appropriate:
                    # Enhance content with cultural context
                    cultural_prompt = self._apply_cultural_integration(state, "response_enhancement")
                    state["response_confidence"] = max(0.3, state.get("response_confidence", 0.5) - 0.1)
                    state["response_reason"] += " (cultural appropriateness enhanced)"
                else:
                    state["response_confidence"] = min(1.0, state.get("response_confidence", 0.5) + 0.1)
                    state["response_reason"] += " (culturally appropriate)"
            
            processing_time = time.time() - start_time
            
            # Add final metadata
            state = add_processing_metadata(
                state,
                "response_generation",
                state.get("response_confidence", 0.5),
                state.get("response_reason", "Final response generated"),
                [],
                processing_time
            )
            
            logger.info("✅ Final response generated")
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            add_error(state, f"Response generation error: {str(e)}")
            state["generated_content"] = "I'm here to support you. How can I help you today?"
        
        return state


