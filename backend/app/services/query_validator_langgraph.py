"""
LangGraph-based Query Validator Service

This service uses LangGraph to validate and categorize user queries
using LLM-based classification instead of keyword matching.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain.schema import HumanMessage, SystemMessage

from .langgraph_state import (
    QueryValidationState, QueryClassification, CrisisAssessment,
    QuerySuggestions, ConversationContext, ConversationMemory, EmotionDetection,
    create_initial_state, update_state_with_classification,
    update_state_with_crisis_assessment, update_state_with_suggestions,
    mark_workflow_complete, DEFAULT_CONFIG, QueryType, CrisisSeverity,
    RoutingPriority
)
from ..prompts.query_classification_prompts import QueryClassificationPrompts
from ..prompts.safety_prompts import SafetyPrompts
from ..prompts.cultural_context_prompts import CulturalContextPrompts
from ..prompts.response_approach_prompts import ResponseApproachPrompts


class LangGraphQueryValidator:
    """
    LangGraph-based query validator that uses LLM for classification.
    """

    def __init__(self, llm_provider=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LangGraph query validator.

        Args:
            llm_provider: LLM provider for making classification calls
            config: Configuration for the LangGraph workflow
        """
        self.llm_provider = llm_provider
        self.config = config or DEFAULT_CONFIG
        self.is_initialized = llm_provider is not None

    async def validate_query(self, query: str, user_id: Optional[str] = None,
                           conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Validate and categorize a user query using LangGraph workflow.

        Args:
            query: The user's query string
            user_id: Optional user identifier
            conversation_history: Optional conversation history

        Returns:
            Dictionary containing validation results and suggestions
        """
        if not self.is_initialized:
            return self._get_fallback_result(query, "LLM provider not initialized")

        # Create initial state
        state = create_initial_state(query, user_id, conversation_history)

        try:
            # Execute LangGraph workflow
            final_state = await self._execute_workflow(state)

            # Extract results
            return self._extract_results(final_state)

        except Exception as e:
            print(f"Error in LangGraph workflow: {e}")
            return self._get_fallback_result(query, str(e))

    async def _execute_workflow(self, state: QueryValidationState) -> QueryValidationState:
        """
        Execute the LangGraph workflow for query validation.

        Args:
            state: Initial workflow state

        Returns:
            Final workflow state
        """
        # Step 1: Classify the query
        classification = await self._classify_query(state["query"])
        state = update_state_with_classification(state, classification)

        # Step 2: Assess for crisis
        if self.config["enable_crisis_detection"]:
            crisis_assessment = await self._assess_crisis(state["query"])
            state = update_state_with_crisis_assessment(state, crisis_assessment)

        # Step 3: Detect emotions (only for non-random queries)
        classification = state.get("classification", {})
        query_type = classification.get("query_type", QueryType.UNCLEAR) if isinstance(classification, dict) else QueryType.UNCLEAR

        # Only detect emotions for mental support and crisis queries
        if query_type in [QueryType.MENTAL_SUPPORT, QueryType.CRISIS]:
            emotion_detection = await self._detect_emotion(state["query"])
            state["emotion_detection"] = emotion_detection
            state["processing_steps_completed"].append("detect_emotion")

        # Step 4: Generate suggestions
        suggestions = await self._generate_suggestions(state)
        state = update_state_with_suggestions(state, suggestions)

        # Step 4: Analyze conversation context (if history available)
        if self.config["enable_context_analysis"] and state["conversation_history"]:
            context = await self._analyze_context(state)
            state["conversation_context"] = context

        # Step 5: Manage memory (if enabled)
        if self.config["enable_memory"]:
            memory = await self._manage_memory(state)
            state["memory_management"] = memory

        # Step 6: Make final routing decision
        routing_decision = self._make_routing_decision(dict(state))
        state["routing_decision"] = routing_decision

        # Step 7: Determine if conversation should proceed
        should_proceed = self._determine_conversation_proceeding(dict(state))
        state["should_proceed_to_conversation"] = should_proceed

        # Mark workflow as complete
        final_response = self._generate_final_response(dict(state))
        state = mark_workflow_complete(state, final_response, routing_decision)

        return state

    async def _classify_query(self, query: str) -> QueryClassification:
        """
        Classify the query using LLM.

        Args:
            query: Query to classify

        Returns:
            QueryClassification result
        """
        if not self.llm_provider:
            return QueryClassification(
                query_type=QueryType.UNCLEAR,
                confidence=0.5,
                reasoning="LLM provider not available",
                keywords_found=[],
                is_crisis=False,
                crisis_severity=None,
                requires_human_intervention=True
            )

        system_prompt = QueryClassificationPrompts.get_query_classification_prompt()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f'Query to classify: "{query}"')
        ]

        response = await self.llm_provider.generate_response(messages)
        parsed_result = QueryClassificationPrompts.parse_classification_response(response)

        # Map string values to enum values
        query_type_str = parsed_result.get("query_type", "UNCLEAR")
        query_type_enum = {
            "MENTAL_SUPPORT": QueryType.MENTAL_SUPPORT,
            "RANDOM_QUESTION": QueryType.RANDOM_QUESTION,
            "UNCLEAR": QueryType.UNCLEAR,
            "CRISIS": QueryType.CRISIS
        }.get(query_type_str, QueryType.UNCLEAR)

        crisis_severity_str = parsed_result.get("crisis_severity")
        crisis_severity_enum = None
        if crisis_severity_str:
            crisis_severity_enum = {
                "LOW": CrisisSeverity.LOW,
                "MEDIUM": CrisisSeverity.MEDIUM,
                "HIGH": CrisisSeverity.HIGH,
                "CRITICAL": CrisisSeverity.CRITICAL
            }.get(crisis_severity_str, CrisisSeverity.LOW)

        return QueryClassification(
            query_type=query_type_enum,
            confidence=parsed_result.get("confidence", 0.5),
            reasoning=parsed_result.get("reasoning", "Classification failed"),
            keywords_found=parsed_result.get("keywords_found", []),
            is_crisis=parsed_result.get("is_crisis", False),
            crisis_severity=crisis_severity_enum,
            requires_human_intervention=parsed_result.get("requires_human_intervention", False)
        )

    async def _assess_crisis(self, query: str) -> CrisisAssessment:
        """
        Assess the query for crisis indicators.

        Args:
            query: Query to assess

        Returns:
            CrisisAssessment result
        """
        if not self.llm_provider:
            return CrisisAssessment(
                is_crisis=False,
                confidence=0.5,
                crisis_indicators=[],
                severity=CrisisSeverity.LOW,
                recommended_action="monitor",
                immediate_risks=[]
            )

        system_prompt = QueryClassificationPrompts.get_crisis_detection_prompt()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f'Query to assess for crisis: "{query}"')
        ]

        # Convert messages to proper format for LLM provider
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                # Already a proper message object
                formatted_messages.append(msg)
            else:
                # Convert dict to proper message object
                if msg.get('role') == 'system':
                    formatted_messages.append(SystemMessage(content=msg.get('content', '')))
                elif msg.get('role') == 'user':
                    formatted_messages.append(HumanMessage(content=msg.get('content', '')))
                else:
                    formatted_messages.append(HumanMessage(content=str(msg)))

        response = await self.llm_provider.generate_response(formatted_messages)

        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {}
        except (json.JSONDecodeError, KeyError):
            result = {}

        return CrisisAssessment(
            is_crisis=result.get("is_crisis", False),
            confidence=result.get("confidence", 0.5),
            crisis_indicators=result.get("crisis_indicators", []),
            severity=CrisisSeverity(result.get("severity", "low")),
            recommended_action=result.get("recommended_action", "monitor"),
            immediate_risks=result.get("immediate_risks", [])
        )

    async def _detect_emotion(self, query: str) -> EmotionDetection:
        """
        Detect emotions in the user query using LLM.

        Args:
            query: Query to analyze for emotions

        Returns:
            EmotionDetection result
        """
        if not self.llm_provider:
            return EmotionDetection(
                detected_emotion="neutral",
                confidence=0.5,
                emotion_score={"neutral": 1.0},
                reasoning="LLM provider not available",
                intensity="low",
                context_relevance="medium"
            )

        system_prompt = QueryClassificationPrompts.get_emotion_detection_prompt()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f'Analyze the emotions in this query: "{query}"')
        ]

        # Convert messages to proper format for LLM provider
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                # Already a proper message object
                formatted_messages.append(msg)
            else:
                # Convert dict to proper message object
                if msg.get('role') == 'system':
                    formatted_messages.append(SystemMessage(content=msg.get('content', '')))
                elif msg.get('role') == 'user':
                    formatted_messages.append(HumanMessage(content=msg.get('content', '')))
                else:
                    formatted_messages.append(HumanMessage(content=str(msg)))

        response = await self.llm_provider.generate_response(formatted_messages)

        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {}
        except (json.JSONDecodeError, KeyError):
            result = {}

        # Default emotion scores
        default_scores = {
            "joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0,
            "surprise": 0.0, "disgust": 0.0, "neutral": 1.0, "anxiety": 0.0
        }

        emotion_scores = result.get("emotion_scores", default_scores)
        detected_emotion = result.get("detected_emotion", "neutral")
        confidence = result.get("confidence", 0.5)
        reasoning = result.get("reasoning", "Emotion detection failed")
        intensity = result.get("intensity", "low")
        context_relevance = result.get("context_relevance", "medium")

        return EmotionDetection(
            detected_emotion=detected_emotion,
            confidence=confidence,
            emotion_score=emotion_scores,
            reasoning=reasoning,
            intensity=intensity,
            context_relevance=context_relevance
        )

    async def _generate_suggestions(self, state: QueryValidationState) -> QuerySuggestions:
        """
        Generate suggestions for handling the query.

        Args:
            state: Current workflow state

        Returns:
            QuerySuggestions result
        """
        if not self.llm_provider:
            return QuerySuggestions(
                suggestions=["Use fallback processing", "Request clarification from user"],
                routing_priority=RoutingPriority.MEDIUM,
                requires_human_intervention=True,
                follow_up_questions=["Could you clarify what you need help with?"],
                next_best_action="fallback_processing"
            )

        system_prompt = QueryClassificationPrompts.get_query_suggestions_prompt()

        context = f"""
        Query: {state['query']}
        Classification: {state['classification']['query_type'].value if state['classification'] else 'unknown'}
        Is Crisis: {state['crisis_assessment']['is_crisis'] if state['crisis_assessment'] else False}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f'Generate suggestions for this query:\n{context}')
        ]

        # Convert messages to proper format for LLM provider
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                # Already a proper message object
                formatted_messages.append(msg)
            else:
                # Convert dict to proper message object
                if msg.get('role') == 'system':
                    formatted_messages.append(SystemMessage(content=msg.get('content', '')))
                elif msg.get('role') == 'user':
                    formatted_messages.append(HumanMessage(content=msg.get('content', '')))
                else:
                    formatted_messages.append(HumanMessage(content=str(msg)))

        response = await self.llm_provider.generate_response(formatted_messages)

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {}
        except (json.JSONDecodeError, KeyError):
            result = {}

        return QuerySuggestions(
            suggestions=result.get("suggestions", ["Process query with standard handling"]),
            routing_priority=RoutingPriority(result.get("routing_priority", "medium")),
            requires_human_intervention=result.get("requires_human_intervention", False),
            follow_up_questions=result.get("follow_up_questions", []),
            next_best_action=result.get("next_best_action", "standard_processing")
        )

    async def _analyze_context(self, state: QueryValidationState) -> ConversationContext:
        """
        Analyze conversation context.

        Args:
            state: Current workflow state

        Returns:
            ConversationContext result
        """
        if not self.llm_provider:
            return ConversationContext(
                conversation_type="new",
                emotional_progression="unclear",
                key_themes=[],
                user_preferences=[],
                suggested_focus="general_support",
                memory_cues=[]
            )

        system_prompt = ResponseApproachPrompts.get_conversation_context_prompt()

        context_text = "\n".join([
            f"{msg['role']}: {msg['text']}"
            for msg in state["conversation_history"][-5:]  # Last 5 messages
        ])

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f'Analyze this conversation context:\n{context_text}')
        ]

        # Convert messages to proper format for LLM provider
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                # Already a proper message object
                formatted_messages.append(msg)
            else:
                # Convert dict to proper message object
                if msg.get('role') == 'system':
                    formatted_messages.append(SystemMessage(content=msg.get('content', '')))
                elif msg.get('role') == 'user':
                    formatted_messages.append(HumanMessage(content=msg.get('content', '')))
                else:
                    formatted_messages.append(HumanMessage(content=str(msg)))

        response = await self.llm_provider.generate_response(formatted_messages)

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {}
        except (json.JSONDecodeError, KeyError):
            result = {}

        return ConversationContext(
            conversation_type=result.get("conversation_type", "new"),
            emotional_progression=result.get("emotional_progression", "unclear"),
            key_themes=result.get("key_themes", []),
            user_preferences=result.get("user_preferences", []),
            suggested_focus=result.get("suggested_focus", "general_support"),
            memory_cues=result.get("memory_cues", [])
        )

    async def _manage_memory(self, state: QueryValidationState) -> ConversationMemory:
        """
        Manage conversation memory.

        Args:
            state: Current workflow state

        Returns:
            ConversationMemory result
        """
        if not self.llm_provider:
            return ConversationMemory(
                to_remember=[],
                to_forget=[],
                memory_summary="",
                privacy_notes="LLM provider not available"
            )

        system_prompt = ResponseApproachPrompts.get_memory_management_prompt()

        context = f"Query: {state['query']}\nClassification: {state['classification']['query_type'].value if state['classification'] else 'unknown'}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f'Manage memory for this interaction:\n{context}')
        ]

        # Convert messages to proper format for LLM provider
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                # Already a proper message object
                formatted_messages.append(msg)
            else:
                # Convert dict to proper message object
                if msg.get('role') == 'system':
                    formatted_messages.append(SystemMessage(content=msg.get('content', '')))
                elif msg.get('role') == 'user':
                    formatted_messages.append(HumanMessage(content=msg.get('content', '')))
                else:
                    formatted_messages.append(HumanMessage(content=str(msg)))

        response = await self.llm_provider.generate_response(formatted_messages)

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {}
        except (json.JSONDecodeError, KeyError):
            result = {}

        return ConversationMemory(
            to_remember=result.get("to_remember", []),
            to_forget=result.get("to_forget", []),
            memory_summary=result.get("memory_summary", ""),
            privacy_notes=result.get("privacy_notes", "")
        )

    def _make_routing_decision(self, state: Dict[str, Any]) -> str:
        """
        Make final routing decision based on workflow results.

        Args:
            state: Current workflow state

        Returns:
            Routing decision string
        """
        # Crisis situations get highest priority
        crisis_assessment = state.get("crisis_assessment", {})
        if isinstance(crisis_assessment, dict) and crisis_assessment.get("is_crisis", False):
            return "crisis_intervention"

        # Mental support queries - proceed to conversation
        classification = state.get("classification", {})
        if isinstance(classification, dict) and classification.get("query_type") == QueryType.MENTAL_SUPPORT:
            return "mental_health_support"

        # Random questions - do not proceed to conversation
        if isinstance(classification, dict) and classification.get("query_type") == QueryType.RANDOM_QUESTION:
            return "random_question_filtered"

        # Unclear queries - do not proceed to conversation
        if isinstance(classification, dict) and classification.get("query_type") == QueryType.UNCLEAR:
            return "clarification_needed"

        # Default routing
        return "standard_processing"

    def _generate_final_response(self, state: Dict[str, Any]) -> str:
        """
        Generate final response based on workflow results.

        Args:
            state: Current workflow state

        Returns:
            Final response string
        """
        classification = state.get("classification", {})
        crisis_assessment = state.get("crisis_assessment", {})

        query_type = classification.get("query_type", QueryType.UNCLEAR) if isinstance(classification, dict) else QueryType.UNCLEAR
        is_crisis = crisis_assessment.get("is_crisis", False) if isinstance(crisis_assessment, dict) else False

        if is_crisis:
            return "I detect this may be a crisis situation. Please reach out to emergency services at 112 or the mental health helpline at 114 immediately."

        if query_type == QueryType.MENTAL_SUPPORT:
            return "I understand you're reaching out for mental health support. I'm here to listen and help."

        if query_type == QueryType.RANDOM_QUESTION:
            return "I see you have a question. While I'm primarily designed for mental health support, I'll do my best to help."

        if query_type == QueryType.UNCLEAR:
            return "I'm not sure I understand your query. Could you please clarify what you'd like help with?"

        return "I'm here to help. What would you like to talk about?"

    def _determine_conversation_proceeding(self, state: Dict[str, Any]) -> bool:
        """
        Determine if the conversation should proceed based on query classification.

        Args:
            state: Current workflow state

        Returns:
            Boolean indicating if conversation should proceed
        """
        classification = state.get("classification", {})
        crisis_assessment = state.get("crisis_assessment", {})

        query_type = classification.get("query_type", QueryType.UNCLEAR) if isinstance(classification, dict) else QueryType.UNCLEAR
        is_crisis = crisis_assessment.get("is_crisis", False) if isinstance(crisis_assessment, dict) else False

        # Always proceed for crisis situations
        if is_crisis:
            return True

        # Proceed for mental support queries
        if query_type == QueryType.MENTAL_SUPPORT:
            return True

        # Do not proceed for random questions or unclear queries
        return False

    def _extract_results(self, state: QueryValidationState) -> Dict[str, Any]:
        """
        Extract results from the final workflow state.

        Args:
            state: Final workflow state

        Returns:
            Dictionary of results
        """
        # Safely extract classification data
        classification = state.get("classification", {})
        if not isinstance(classification, dict):
            classification = {}

        # Safely extract crisis assessment data
        crisis_assessment = state.get("crisis_assessment", {})
        if not isinstance(crisis_assessment, dict):
            crisis_assessment = {}

        # Safely extract suggestions data
        suggestions = state.get("suggestions", {})
        if not isinstance(suggestions, dict):
            suggestions = {}

        # Safely extract emotion detection data
        emotion_detection = state.get("emotion_detection", {})
        if not isinstance(emotion_detection, dict):
            emotion_detection = {}

        # Convert datetime to ISO string if present
        processing_timestamp = state.get("processing_timestamp")
        if processing_timestamp and hasattr(processing_timestamp, 'isoformat'):
            processing_timestamp = processing_timestamp.isoformat()

        return {
            "query_type": classification.get("query_type", QueryType.UNCLEAR).value,
            "confidence": classification.get("confidence", 0.5),
            "reasoning": classification.get("reasoning", "Classification failed"),
            "keywords_found": classification.get("keywords_found", []),
            "is_crisis": crisis_assessment.get("is_crisis", False),
            "crisis_severity": crisis_assessment.get("severity", "low").value if crisis_assessment.get("severity") else "low",
            "suggestions": suggestions.get("suggestions", []),
            "routing_priority": suggestions.get("routing_priority", "medium").value if suggestions.get("routing_priority") else "medium",
            "requires_human_intervention": suggestions.get("requires_human_intervention", False),
            "routing_decision": state.get("routing_decision", "standard_processing"),
            "final_response": state.get("final_response", "Processing complete"),
            "should_proceed_to_conversation": state.get("should_proceed_to_conversation", False),
            "processing_timestamp": processing_timestamp,
            "emotion_detection": emotion_detection,
            "errors": state.get("errors", [])
        }

    def _get_fallback_result(self, query: str, error: str) -> Dict[str, Any]:
        """
        Get fallback result when workflow fails.

        Args:
            query: Original query
            error: Error message

        Returns:
            Fallback result dictionary
        """
        return {
            "query_type": "unclear",
            "confidence": 0.5,
            "reasoning": f"Workflow failed: {error}",
            "keywords_found": [],
            "is_crisis": False,
            "crisis_severity": "low",
            "suggestions": ["Use fallback processing", "Request clarification from user"],
            "routing_priority": "medium",
            "requires_human_intervention": True,
            "routing_decision": "fallback_processing",
            "final_response": "I'm having trouble processing your request. Could you please rephrase or clarify what you need help with?",
            "should_proceed_to_conversation": False,
            "processing_timestamp": datetime.now().isoformat(),
            "errors": [error]
        }


    async def execute_workflow(self, query: str, user_id: Optional[str] = None,
                             conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Execute the complete LangGraph workflow for query validation and filtering.

        This method provides a clean interface for the workflow that:
        1. Validates the query using LangGraph
        2. Determines if conversation should proceed
        3. Returns structured results

        Args:
            query: The user's query string
            user_id: Optional user identifier
            conversation_history: Optional conversation history

        Returns:
            Dictionary containing workflow results including should_proceed_to_conversation flag
        """
        if not self.is_initialized:
            return self._get_fallback_result(query, "LLM provider not initialized")

        # Create initial state
        state = create_initial_state(query, user_id, conversation_history)

        try:
            # Execute LangGraph workflow
            final_state = await self._execute_workflow(state)

            # Extract results
            return self._extract_results(final_state)

        except Exception as e:
            print(f"Error in LangGraph workflow: {e}")
            return self._get_fallback_result(query, str(e))


def initialize_langgraph_query_validator(llm_provider=None) -> LangGraphQueryValidator:
    """
    Initialize and return a LangGraph query validator service instance.

    Args:
        llm_provider: Optional LLM provider

    Returns:
        LangGraphQueryValidator instance
    """
    return LangGraphQueryValidator(llm_provider=llm_provider)