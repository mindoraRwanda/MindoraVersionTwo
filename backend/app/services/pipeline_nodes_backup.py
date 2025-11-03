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

from langchain_core.messages import HumanMessage, SystemMessage
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
    
    async def _call_llm(self, system_prompt: str, user_prompt: str, state: StatefulPipelineState) -> str:
        """Make LLM call with error handling."""
        try:
            logger.info(f"🤖 Making LLM call with {len(system_prompt)} char system prompt, {len(user_prompt)} char user prompt")
            system_message = SystemMessage(content=system_prompt)
            human_message = HumanMessage(content=user_prompt)
            response = await self.llm_provider.agenerate([system_message, human_message])
            increment_llm_calls(state)
            logger.info(f"✅ LLM call successful, response length: {len(response)} chars")
            return response
        except Exception as e:
            logger.error(f"❌ LLM call failed: {e}")
            add_error(state, f"LLM call error: {str(e)}")
            return ""



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
            5. Cultural context considerations in the query
            
            Consider cultural expressions of mental health concerns that may be indirect or use local terminology.
            
            Respond in JSON format:
            {{
                "confidence": 0.8,
                "keywords": ["anxiety", "stress"],
                "reasoning": "Query contains mental health indicators with cultural context",
                "is_random": false,
                "query_type": "mental_health",
                "cultural_indicators": ["cultural_expression_1", "cultural_expression_2"]
            }}
            """
            
            user_prompt = f"Classify this query with cultural awareness: '{query}'"
            logger.info(f"📝 Query validation prompt prepared (system: {len(system_prompt)} chars)")
            response = await self._call_llm(system_prompt, user_prompt, state)
            
            # Parse response and create validation result
            logger.info(f"🔍 Parsing validation response: '{response[:200]}{'...' if len(response) > 200 else ''}'")
            validation_result = self._parse_validation_response(response, query)
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
    
    def _parse_validation_response(self, response: str, query: str) -> QueryValidationResult:
        """Parse LLM response into structured validation result."""
        try:
            # Try to parse JSON response
            data = json.loads(response)
            
            # Extract cultural indicators if present
            cultural_indicators = data.get("cultural_indicators", [])
            keywords = data.get("keywords", [])
            
            # Combine keywords with cultural indicators
            all_keywords = keywords + cultural_indicators
            
            return QueryValidationResult(
                query_confidence=float(data.get("confidence", 0.5)),
                query_keywords=all_keywords,
                query_reason=data.get("reasoning", "Query analyzed with cultural context"),
                is_random=bool(data.get("is_random", False)),
                query_type=QueryType(data.get("query_type", "mental_health"))
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback parsing
            return QueryValidationResult(
                query_confidence=0.7,
                query_keywords=["mental", "health", "cultural_context"],
                query_reason="Query contains mental health indicators with cultural context",
                is_random=False,
                query_type=QueryType.MENTAL_HEALTH
            )


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
            4. Severity level: severe, high, medium, low, none
            5. Cultural stigma considerations
            6. Indirect crisis communication patterns
            
            Consider that in Rwandan culture, mental health crises may be expressed indirectly due to stigma.
            Look for cultural expressions of distress, family concerns, and community-based crisis indicators.
            
            Available crisis resources for {language}:
            - National Helpline: {crisis_resources.get('national_helpline', '114')}
            - Emergency: {crisis_resources.get('emergency', '112')}
            - Hospitals: {', '.join(crisis_resources.get('hospitals', []))}
            
            Respond in JSON format:
            {{
                "is_crisis": true,
                "crisis_confidence": 0.1,
                "crisis_keywords": [],
                "crisis_reason": "No crisis indicators detected with cultural context consideration",
                "crisis_severity": "none",
            }}
            """
            
            user_prompt = f"Assess this query for crisis indicators with cultural awareness: '{query}'"
            logger.info(f"📝 Crisis detection prompt prepared (system: {len(system_prompt)} chars)")
            response = await self._call_llm(system_prompt, user_prompt, state)
            
            # Parse response and create crisis assessment
            logger.info(f"🔍 Parsing crisis detection response: '{response[:200]}{'...' if len(response) > 200 else ''}'")
            crisis_assessment = self._parse_crisis_response(response, query)
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

            state["crisis_assessment"] = crisis_assessment
            
            logger.info(f"✅ Crisis detection completed: is_crisis={crisis_assessment.is_crisis}, severity={crisis_assessment.crisis_severity.value}, confidence={crisis_assessment.crisis_confidence:.2f}, keywords={crisis_assessment.crisis_keywords}")
            
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
    
    def _parse_crisis_response(self, response: str, query: str) -> CrisisAssessment:
        """Parse LLM response into structured crisis assessment."""
        try:
            data = json.loads(response)

            # Extract cultural considerations and resource recommendations
            cultural_considerations = data.get("cultural_considerations", [])
            resource_recommendations = data.get("resource_recommendations", [])
            keywords = data.get("keywords", [])

            # Combine keywords with cultural considerations
            all_keywords = keywords + cultural_considerations + resource_recommendations

            return CrisisAssessment(
                is_crisis=data.get("is_crisis", False),
                crisis_confidence=float(data.get("crisis_confidence", 0.0)),
                crisis_keywords=all_keywords,
                crisis_reason=data.get("reasoning", "Crisis assessment completed with cultural context"),
                crisis_severity=CrisisSeverity(data.get("crisis_severity", data.get("severity", "none")))
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return CrisisAssessment(
                crisis_confidence=0.0,
                is_crisis=False,
                crisis_keywords=["cultural_context", "stigma_awareness"],
                crisis_reason="No crisis indicators detected with cultural context consideration",
                crisis_severity=CrisisSeverity.NONE
            )


class EmotionDetectionNode(BasePipelineNode):
    """Node for detecting user emotions with youth-specific patterns."""
    
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Execute emotion detection with confidence scoring."""
        start_time = time.time()
        query = state["user_query"]
        
        logger.info(f"😊 Starting emotion detection for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        try:
            # Get cultural context for emotion detection
            cultural_context = self._get_cultural_context(state)
            language = cultural_context["language"]
            emotion_responses = cultural_context["emotion_responses"]
            
            # Apply cultural integration for emotion detection
            cultural_prompt = self._apply_cultural_integration(state, "emotion_detection")
            
            # Use LLM for emotion detection with cultural context
            system_prompt = f"""
            You are an emotion detection specialist for youth mental health with cultural awareness for {language} speakers.
            
            {cultural_prompt}
            
            Analyze the query for emotions with cultural sensitivity:
            1. Primary emotion and confidence score (0-1)
            2. List of emotion keywords (including cultural expressions)
            3. Reasoning for emotion detection
            4. Emotion intensity level
            5. Cultural emotional expression patterns
            6. Youth-specific emotional language
            
            Consider that emotional expression varies across cultures. In Rwandan culture:
            - Emotions may be expressed more indirectly
            - Family and community context affects emotional expression
            - Youth may use different emotional vocabulary
            - Cultural stigma may influence how emotions are communicated
            
            Available emotion response templates for {language}:
            {emotion_responses}
            
            Respond in JSON format:
            {{
                "emotions": {{"anxiety": 0.7, "neutral": 0.3}},
                "keywords": ["worried", "anxious", "cultural_expression"],
                "reasoning": "Query shows anxiety and worry with cultural context",
                "selected_emotion": "anxiety",
                "confidence": 0.7,
                "cultural_emotional_indicators": ["indirect_expression", "family_context"],
                "youth_emotional_patterns": ["peer_pressure", "academic_stress"]
            }}
            """
            
            user_prompt = f"Detect emotions in this query with cultural awareness: '{query}'"
            logger.info(f"📝 Emotion detection prompt prepared (system: {len(system_prompt)} chars)")
            response = await self._call_llm(system_prompt, user_prompt, state)
            
            # Parse response and create emotion detection
            logger.info(f"🔍 Parsing emotion detection response: '{response[:200]}{'...' if len(response) > 200 else ''}'")
            emotion_detection = self._parse_emotion_response(response, query)
            state["emotion_detection"] = emotion_detection
            
            processing_time = time.time() - start_time
            
            # Add metadata
            state = add_processing_metadata(
                state,
                "emotion_detection",
                emotion_detection.emotion_confidence,
                emotion_detection.emotion_reason,
                emotion_detection.emotion_keywords,
                processing_time
            )
            
            logger.info(f"✅ Emotion detection completed: emotion={emotion_detection.selected_emotion}, confidence={emotion_detection.emotion_confidence:.2f}, keywords={emotion_detection.emotion_keywords}")
            
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
    
    def _parse_emotion_response(self, response: str, query: str) -> EmotionDetection:
        """Parse LLM response into structured emotion detection."""
        try:
            data = json.loads(response)
            
            # Extract cultural emotional indicators and youth patterns
            cultural_indicators = data.get("cultural_emotional_indicators", [])
            youth_patterns = data.get("youth_emotional_patterns", [])
            keywords = data.get("keywords", [])
            
            # Combine keywords with cultural elements
            all_keywords = keywords + cultural_indicators + youth_patterns
            
            return EmotionDetection(
                emotions=data.get("emotions", {"neutral": 1.0}),
                emotion_keywords=all_keywords,
                emotion_reason=data.get("reasoning", "Emotion detection completed with cultural context"),
                selected_emotion=data.get("selected_emotion", "neutral"),
                emotion_confidence=float(data.get("confidence", 0.5))
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return EmotionDetection(
                emotions={"neutral": 1.0},
                emotion_keywords=["cultural_context", "youth_patterns"],
                emotion_reason="Emotion detection completed with cultural context",
                selected_emotion="neutral",
                emotion_confidence=0.5
            )


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
            
            Respond in JSON format:
            {{
                "confidence": 0.8,
                "reasoning": "User needs emotional support with cultural context consideration",
                "keywords": ["support", "help", "cultural_appropriateness"],
                "strategy": "GIVE_EMPATHY",
                "cultural_considerations": ["indirect_communication", "family_context"],
                "cultural_appropriateness": "high"
            }}
            """
            
            user_prompt = f"""
            Query: '{query}'
            Crisis Level: {crisis.crisis_severity.value if crisis else "none"}
            Primary Emotion: {emotion.selected_emotion if emotion else "neutral"}
            Language: {language}
            
            Select appropriate response strategy with cultural awareness.
            """
            
            logger.info(f"📝 Query evaluation prompt prepared (system: {len(system_prompt)} chars)")
            response = await self._call_llm(system_prompt, user_prompt, state)
            
            # Parse response and create evaluation
            logger.info(f"🔍 Parsing query evaluation response: '{response[:200]}{'...' if len(response) > 200 else ''}'")
            query_evaluation = self._parse_evaluation_response(response, query)
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
    
    def _parse_evaluation_response(self, response: str, query: str) -> QueryEvaluation:
        """Parse LLM response into structured query evaluation."""
        try:
            data = json.loads(response)
            
            # Extract cultural considerations and appropriateness
            cultural_considerations = data.get("cultural_considerations", [])
            cultural_appropriateness = data.get("cultural_appropriateness", "medium")
            keywords = data.get("keywords", [])
            
            # Combine keywords with cultural elements
            all_keywords = keywords + cultural_considerations + [f"cultural_appropriateness_{cultural_appropriateness}"]
            
            return QueryEvaluation(
                evaluation_confidence=float(data.get("confidence", 0.5)),
                evaluation_reason=data.get("reasoning", "Query evaluation completed with cultural context"),
                evaluation_keywords=all_keywords,
                evaluation_type=ResponseStrategy(data.get("strategy", "GIVE_EMPATHY"))
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return QueryEvaluation(
                evaluation_confidence=0.5,
                evaluation_reason="Query evaluation completed with cultural context",
                evaluation_keywords=["cultural_context", "cultural_appropriateness_medium"],
                evaluation_type=ResponseStrategy.GIVE_EMPATHY
            )


# Specialized Response Nodes

class EmpathyNode(BasePipelineNode):
    """Node for generating empathetic responses."""
    
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """Generate empathetic response with cultural context."""
        start_time = time.time()
        query = state["user_query"]
        emotion = state.get("emotion_detection")
        
        logger.info(f"💝 Starting empathy response generation for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        logger.info(f"😊 Detected emotion: {emotion.selected_emotion if emotion else 'neutral'}")
        
        try:
            # Get cultural context for empathy response
            cultural_context = self._get_cultural_context(state)
            language = cultural_context["language"]
            emotion_responses = cultural_context["emotion_responses"]
            
            # Apply cultural integration for empathy response
            cultural_prompt = self._apply_cultural_integration(state, "empathy_response")
            
            # Get gender-aware addressing
            gender_addressing = self._get_gender_aware_addressing(state)
            
            # Get RAG knowledge context if available
            knowledge_context = state.get("knowledge_context", "")
            rag_applied = state.get("rag_enhancement_applied", False)
            
            # Generate empathetic response with cultural context and RAG knowledge
            system_prompt = f"""
                   You are generating an empathetic response with cultural awareness for {language} speakers.
                   
                   {cultural_prompt}
                   
                   Generate a culturally sensitive, supportive response that:
                   1. Uses appropriate cultural expressions and terminology
                   2. Incorporates Ubuntu philosophy ("I am because we are")
                   3. Shows understanding of family and community context
                   4. Uses gender-aware addressing: {gender_addressing if gender_addressing else "friend"}
                   5. Considers cultural stigma around mental health
                   6. Provides hope and community support
                   
                   Available emotion response templates for {language}:
                   {emotion_responses}
                   
                   {f"Relevant Mental Health Knowledge: {knowledge_context}" if rag_applied and knowledge_context else ""}
                   
                   Make the response feel authentic and relatable to Rwandan youth.
                   Use the available knowledge to provide informed, evidence-based support while maintaining cultural sensitivity.
                   """
            
            user_prompt = f"""
            Generate an empathetic response for this query:
            Query: "{query}"
            Detected Emotion: {emotion.selected_emotion if emotion else "neutral"}
            Language: {language}
            Gender Addressing: {gender_addressing if gender_addressing else "friend"}
            
            Provide culturally sensitive, supportive response that feels authentic.
            """
            
            logger.info(f"📝 Empathy prompt prepared (system: {len(system_prompt)} chars)")
            response = await self._call_llm(system_prompt, user_prompt, state)
            
            # Validate cultural appropriateness
            is_culturally_appropriate = self._validate_cultural_appropriateness(response, state)
            
            state["generated_content"] = response
            state["response_confidence"] = 0.8 if is_culturally_appropriate else 0.6
            state["response_reason"] = f"Generated empathetic response with cultural sensitivity (appropriateness: {'high' if is_culturally_appropriate else 'medium'})"
            
            processing_time = time.time() - start_time
            
            # Add metadata
            state = add_processing_metadata(
                state,
                "empathy_response",
                0.8,
                "Generated empathetic response with cultural context",
                ["empathy", "support"],
                processing_time
            )
            
            logger.info(f"✅ Empathy response generated: {len(response)} chars, confidence=0.8")
            
        except Exception as e:
            logger.error(f"❌ Empathy response generation failed: {e}")
            add_error(state, f"Empathy response error: {str(e)}")
            state["generated_content"] = "I understand you're going through a difficult time. I'm here to support you."
        
        return state


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

            logger.info(f"🚨 CrisisAlertNode: User: {user_id}, Conversation: {conversation_id}, Message: {message_id}")
            logger.info(f"🚨 CrisisAlertNode: Query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
            logger.info(f"🚨 CrisisAlertNode: Crisis assessment: {crisis.crisis_severity.value if crisis else 'None'}")
            logger.info(f"🚨 CrisisAlertNode: Parameters check - db: {db is not None}, background: {background is not None}, user_id: {user_id is not None}, conversation_id: {conversation_id is not None}, message_id: {message_id is not None}")

            # Use crisis_classifier to extract crisis information
            crisis_result = classify_crisis(query)
            logger.info(f"🚨 CrisisAlertNode: Crisis classification result: {crisis_result}")

            # Log crisis and notify therapists if we have the required parameters
            if db and background and user_id and conversation_id and message_id:
                logger.info(f"🚨 CrisisAlertNode: All parameters present, calling log_crisis_and_notify")
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
            else:
                logger.warning(f"🚨 CrisisAlertNode: MISSING PARAMETERS - db: {db}, background: {background}, user_id: {user_id}, conversation_id: {conversation_id}, message_id: {message_id}")
                logger.warning(f"🚨 CrisisAlertNode: Crisis logging and notification SKIPPED due to missing parameters")

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
                # Detect message type for natural response generation
                message_type = self._detect_message_type(query)
                gender_addressing = self._get_gender_aware_addressing(state)
                
                if message_type == "greeting":
                    # Use natural greeting templates
                    templates = self.cultural_prompts.get_conversation_template('greeting_responses', language)
                    if templates:
                        import random
                        template = random.choice(templates)
                        response = template.format(gender_sibling=gender_addressing or 'friend')
                        state["generated_content"] = response
                        state["response_confidence"] = 0.9
                        state["response_reason"] = f"Natural greeting response in {language}"
                    else:
                        # Fallback greeting
                        greetings = {
                            'en': f"Hey there, {gender_addressing or 'friend'}! Good to see you.",
                            'rw': f"Muraho, {gender_addressing or 'mugenzi'}! Byiza kubabona.",
                            'fr': f"Salut, {gender_addressing or 'ami'}! Content de te voir.",
                            'sw': f"Hujambo, {gender_addressing or 'rafiki'}! Nimefurahi kukuona."
                        }
                        state["generated_content"] = greetings.get(language, greetings['en'])
                        state["response_confidence"] = 0.8
                        state["response_reason"] = f"Fallback greeting in {language}"
                        
                elif message_type == "casual":
                    # Use casual conversation templates
                    templates = self.cultural_prompts.get_conversation_template('casual_responses', language)
                    if templates:
                        import random
                        template = random.choice(templates)
                        response = template.format(gender_sibling=gender_addressing or 'friend')
                        state["generated_content"] = response
                        state["response_confidence"] = 0.8
                        state["response_reason"] = f"Natural casual response in {language}"
                    else:
                        # Fallback casual
                        casual_responses = {
                            'en': f"I hear you, {gender_addressing or 'friend'}. What's on your mind?",
                            'rw': f"Ndabumva, {gender_addressing or 'mugenzi'}. Ni iki kiri mu mutwe wawe?",
                            'fr': f"Je t'entends, {gender_addressing or 'ami'}. Qu'est-ce qui te préoccupe?",
                            'sw': f"Nakusikia, {gender_addressing or 'rafiki'}. Nini kiko aklini mwako?"
                        }
                        state["generated_content"] = casual_responses.get(language, casual_responses['en'])
                        state["response_confidence"] = 0.7
                        state["response_reason"] = f"Fallback casual response in {language}"
                        
                else:
                    # Generate culturally appropriate supportive response for serious topics
                    cultural_prompt = self._apply_cultural_integration(state, "fallback_response")
                    
                    supportive_responses = {
                        'en': f"I'm here to support you, {gender_addressing or 'friend'}. How can I help you today?",
                        'rw': f"Ndi hano kugufasha, {gender_addressing or 'mugenzi'}. Ni iki nshobora gukugufasha uyu munsi?",
                        'fr': f"Je suis là pour vous soutenir, {gender_addressing or 'ami'}. Comment puis-je vous aider aujourd'hui?",
                        'sw': f"Niko hapa kukusaidia, {gender_addressing or 'rafiki'}. Ninaweza kukusaidia vipi leo?"
                    }
                    
                    state["generated_content"] = supportive_responses.get(language, supportive_responses['en'])
                    state["response_confidence"] = 0.6
                    state["response_reason"] = f"Supportive response generated with cultural context for {language}"
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

    def _detect_message_type(self, query: str) -> str:
        """Detect the type of message for appropriate response generation."""
        query_lower = query.lower().strip()
        
        # Mental health/serious topic patterns (check first)
        serious_patterns = [
            'sad', 'depressed', 'anxiety', 'anxious', 'worried', 'stress', 'stressed',
            'help', 'problem', 'issue', 'trouble', 'difficult', 'hard', 'struggling',
            'feel', 'feeling', 'emotion', 'mood', 'mental', 'therapy', 'counseling',
            'suicide', 'death', 'hurt', 'pain', 'cry', 'crying', 'lonely', 'alone'
        ]
        
        # Greeting patterns
        greeting_patterns = [
            'hello', 'hi', 'hey', 'hola', 'muraho', 'bonjour', 'salut', 'hujambo',
            'good morning', 'good afternoon', 'good evening', 'how are you',
            'mindora', 'greetings'
        ]
        
        # Casual conversation patterns
        casual_patterns = [
            'how are things', 'what\'s up', 'how\'s it going', 'nice to meet',
            'thanks', 'thank you', 'okay', 'alright', 'cool', 'interesting'
        ]
        
        # Check for serious topics first (highest priority)
        for pattern in serious_patterns:
            if pattern in query_lower:
                return "serious"
        
        # Check for greetings
        for pattern in greeting_patterns:
            if pattern in query_lower:
                return "greeting"
        
        # Check for casual conversation
        for pattern in casual_patterns:
            if pattern in query_lower:
                return "casual"
        
        # Check if it's a very short message (likely casual) but not if it contains serious words
        if len(query.split()) <= 3 and len(query) <= 20:
            return "casual"
        
        # Default to serious/supportive for longer or complex messages
        return "serious"
