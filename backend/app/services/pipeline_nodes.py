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
    QueryType, CrisisSeverity, ResponseStrategy,
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

            # For response-generation calls (no structured schema), inject the last
            # few conversation turns so the model can give context-aware replies.
            # Structured-output calls (classification/analysis) stay as [system, human]
            # to avoid polluting the JSON schema response.
            messages: List[Any] = [system_message]
            if not structured_output:
                history = state.get("conversation_history") or []
                for turn in history[-8:]:  # keep last 4 exchanges (8 messages)
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
            You are a crisis detection specialist with cultural awareness for {language} speakers.
            
            {cultural_prompt}
            
            Analyze the query for crisis indicators with cultural sensitivity:
            0. Is crisis present? (true/false)
            1. Crisis confidence score (0-1)
            2. Crisis keywords found (including cultural expressions)
            3. Reasoning for assessment
            4. crisis_severity level: severe, high, medium, low, none
            5. Cultural stigma considerations
            6. Indirect crisis communication patterns
            
            Consider that in Rwandan culture, mental health crises may be expressed indirectly due to stigma.
            Look for cultural expressions of distress, family concerns, and community-based crisis indicators.
            
            Available crisis resources for {language}:
            - National Helpline: {crisis_resources.get('national_helpline', '114')}
            - Emergency: {crisis_resources.get('emergency', '112')}
            - Hospitals: {', '.join(crisis_resources.get('hospitals', []))}
            """
            
            user_prompt = f"Assess this query for crisis indicators with cultural awareness: '{query}'"
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
            
            processing_time = time.time() - start_time
            
            # Add metadata
            add_processing_metadata(
                state, 
                "emotion_detection", 
                result['confidence'], 
                reasoning, 
                result['cultural_markers'], 
                processing_time
            )
            
            logger.info(f"✅ Emotion detected: {result['selected_emotion']} ({result['confidence']:.2f})")

            
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
    



# Specialized Response Nodes

class EmpathyNode(BasePipelineNode):
    """Node for generating empathetic responses."""
    
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Generate empathetic response with cultural context."""
        system_prompt_template = """
            You are generating an empathetic response with cultural awareness for {language} speakers.
            
            {cultural_prompt}
            
            Generate a culturally sensitive, supportive response that:
            1. Uses appropriate cultural expressions and terminology
            2. Incorporates Ubuntu philosophy ("I am because we are")
            3. Shows understanding of family and community context
            4. Uses gender-aware addressing: {gender_addressing}
            5. Considers cultural stigma around mental health
            6. Provides hope and community support
            
            Available emotion response templates for {language}:
            {emotion_responses}
            
            {knowledge_context}
            
            Make the response feel authentic and relatable to Rwandan youth.
            Keep your response concise and to the point, ideally under 3 sentences.
            Use the available knowledge to provide informed, evidence-based support while maintaining cultural sensitivity.
            """
        
        user_prompt_template = """
            Generate an empathetic response for this query:
            Query: "{query}"
            Detected Emotion: {emotion}
            Language: {language}
            Gender Addressing: {gender_addressing}
            
            Provide culturally sensitive, supportive response that feels authentic.
            """
        
        return await self._generate_response(state, "empathy", system_prompt_template, user_prompt_template)


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
            
            # Generate elaboration questions with cultural context
            system_prompt = f"""
            You are a mental health support specialist with cultural awareness for {language} speakers.
            
            {cultural_prompt}
            
            Generate gentle, culturally appropriate questions to encourage the user to share more details:
            1. Use culturally sensitive language and expressions
            2. Consider indirect communication styles common in Rwandan culture
            3. Show respect for family and community context
            4. Use gender-aware addressing: {gender_addressing if gender_addressing else "friend"}
            5. Avoid direct or intrusive questioning
            6. Incorporate Ubuntu philosophy of community support
            
            Make questions feel natural and culturally appropriate for Rwandan youth.
            Keep your response concise and to the point, ideally under 3 sentences.
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
            
            system_prompt = """
            You are a mental health support specialist. The user's query seems contradictory or confusing.
            Generate gentle questions to help clarify their situation and provide better support.
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
                   You are a mental health support specialist with cultural awareness for {language} speakers.
                   
                   {cultural_prompt}
                   
                   Provide culturally appropriate coping strategies and suggestions:
                   1. Use culturally relevant examples and references
                   2. Consider family and community support systems
                   3. Incorporate traditional and modern approaches
                   4. Use gender-aware addressing: {gender_addressing if gender_addressing else "friend"}
                   5. Focus on practical, actionable advice
                   6. Consider cultural stigma and accessibility
                   7. Emphasize Ubuntu philosophy of community support
                   
                   {f"Relevant Mental Health Knowledge: {knowledge_context}" if rag_applied and knowledge_context else ""}
                   
                   Make suggestions feel authentic and culturally appropriate for Rwandan youth.
                   Keep your response concise and to the point, ideally under 3 sentences.
                   Use the available knowledge to provide evidence-based, effective coping strategies.
                   """
            
            user_prompt = f"""
            Provide coping strategies for this query:
            Query: "{query}"
            Emotion: {emotion.selected_emotion if emotion else "neutral"}
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
            
            system_prompt = """
            You are a mental health support specialist. Provide step-by-step guidance and support.
            Break down complex situations into manageable steps and offer ongoing support.
            Keep your response concise and to the point, ideally under 3 sentences.
            """
            
            user_prompt = f"Provide step-by-step guidance for this query: '{query}'"
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
            You are a crisis intervention specialist with cultural awareness for {language} speakers.

            {cultural_prompt}

            Generate an immediate, culturally appropriate response for someone in crisis:
            1. Use culturally sensitive language and expressions
            2. Show compassion while being direct about safety
            3. Use gender-aware addressing: {gender_addressing if gender_addressing else "friend"}
            4. Include culturally appropriate emergency resources
            5. Consider family and community support systems
            6. Emphasize Ubuntu philosophy of community care
            7. Address cultural stigma around mental health crises

            Available crisis resources for {language}:
            - National Helpline: {crisis_resources.get('national_helpline', '114')}
            - Emergency: {crisis_resources.get('emergency', '112')}
            - Hospitals: {', '.join(crisis_resources.get('hospitals', []))}
            - Community Health: {crisis_resources.get('community_health', 'Contact local health center')}

            Make the response feel supportive and culturally appropriate while prioritizing safety.
            Keep your response concise and to the point, ideally under 3 sentences, but ensure safety information is clear.
            """

            user_prompt = f"""
            Generate crisis response for this query:
            Query: '{query}'
            Crisis severity: {crisis.crisis_severity.value if crisis else "unknown"}
            Language: {language}
            Gender Addressing: {gender_addressing if gender_addressing else "friend"}

            Provide culturally sensitive crisis intervention response.
            """

            response = await self._call_llm(system_prompt, user_prompt, state)

            # Add emergency resources
            emergency_response = f"""
                {response}

                🚨 IMMEDIATE SUPPORT RESOURCES:
                • Rwanda Mental Health Hotline: 116 (free, 24/7)
                • Emergency Services: 112
                • National Suicide Prevention: 116
                • Crisis Text Line: Text HOME to 741741

                You are not alone. Please reach out to these resources immediately.
            """

            state["generated_content"] = emergency_response
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


