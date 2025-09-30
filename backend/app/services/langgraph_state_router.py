"""
LLM-Enhanced State Router with Intelligent Decision Making

This module enhances the stateful conversation system with LLM-powered analysis
for intelligent state transitions and response generation, without complex LangGraph workflows.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain.schema import HumanMessage, SystemMessage

from backend.app.services.session_state_manager import (
    SessionStateManager, ConversationState, CrisisSeverity, SessionState, session_manager
)
from backend.app.services.crisis_interceptor import crisis_interceptor
from backend.app.services.llm_service import LLMService


class ConversationContext:
    """Context for conversation state decisions"""
    def __init__(self, session_id: str, current_state: str, conversation_history: List[Dict[str, Any]],
                 user_message: str, emotion_data: Optional[Dict[str, Any]] = None,
                 cultural_context: Optional[Dict[str, Any]] = None, crisis_flags: Optional[List[str]] = None,
                 message_count: int = 0, query_validation: Optional[Dict[str, Any]] = None):
        self.session_id = session_id
        self.current_state = current_state
        self.conversation_history = conversation_history
        self.user_message = user_message
        self.emotion_data = emotion_data or {}
        self.cultural_context = cultural_context or {}
        self.crisis_flags = crisis_flags or []
        self.message_count = message_count
        self.query_validation = query_validation or {}


class StateDecision:
    """LLM decision for state transition"""
    def __init__(self, next_state: str, confidence: float, reasoning: str,
                 suggested_actions: List[str], response_strategy: str,
                 cultural_considerations: Optional[Dict[str, Any]] = None):
        self.next_state = next_state
        self.confidence = confidence
        self.reasoning = reasoning
        self.suggested_actions = suggested_actions
        self.response_strategy = response_strategy
        self.cultural_considerations = cultural_considerations or {}


class LLMEnhancedStateRouter:
    """LLM-enhanced state router with intelligent decision making"""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self.session_manager = session_manager

    async def make_llm_state_decision(self, context: ConversationContext) -> StateDecision:
        """Use LLM to make intelligent state transition decisions"""
        # Check for crisis override first
        is_crisis, crisis_flags, severity = crisis_interceptor.detect_crisis(
            context.user_message, context.emotion_data
        )

        if is_crisis:
            return StateDecision(
                next_state=ConversationState.CRISIS_INTERVENTION.value,
                confidence=1.0,
                reasoning="Crisis detected - immediate intervention required",
                suggested_actions=["crisis_intervention"],
                response_strategy="crisis_protocol"
            )

        # Prepare LLM prompt for state decision
        prompt = self._build_state_decision_prompt(context)

        try:
            # Get LLM decision
            messages = [SystemMessage(content=prompt["system"]),
                       HumanMessage(content=prompt["user"])]

            if not self.llm_service.llm_provider:
                raise Exception("LLM provider not available")

            if not self.llm_service.llm_provider.is_available():
                raise Exception("LLM provider not available")

            response = await self.llm_service.llm_provider.generate_response(messages)

            # Parse LLM response
            return self._parse_llm_state_decision(response)

        except Exception as e:
            print(f"LLM state decision failed: {e}")
            # Fallback to rule-based decision
            return self._fallback_state_decision(context)

    def _build_state_decision_prompt(self, context: ConversationContext) -> Dict[str, str]:
        """Build prompt for LLM state decision"""
        current_state = context.current_state
        user_message = context.user_message
        history = context.conversation_history

        # Build conversation history summary
        recent_messages = history[-5:] if len(history) > 5 else history
        history_summary = "\n".join([
            f"{msg['role']}: {msg.get('content', msg.get('text', ''))[:100]}{'...' if len(msg.get('content', msg.get('text', ''))) > 100 else ''}"
            for msg in recent_messages
        ])

        system_prompt = """You are an expert mental health conversation state manager. Analyze the conversation and determine the most appropriate next state and response strategy.

Available states:
- initial_distress: User expressing initial emotional distress
- awaiting_elaboration: User needs to provide more details
- awaiting_clarification: Need to clarify user's situation
- suggestion_pending: Ready to offer coping strategies
- guiding_in_progress: Actively guiding user through techniques
- conversation_idle: Conversation has become idle
- crisis_intervention: Crisis situation requiring immediate intervention

Response strategy options:
- empathy_building: Focus on validation and trust building
- information_gathering: Ask questions to understand better
- clarification_seeking: Need more specific information
- solution_offering: Provide coping strategies and suggestions
- active_guidance: Step-by-step guidance and practice
- re_engagement: Gently check in and re-engage
- crisis_protocol: Immediate crisis intervention

Consider:
- User's emotional state and needs
- Conversation flow and context
- Cultural sensitivity (Rwandan context)
- Therapeutic best practices
- User's readiness for different types of support

Respond with ONLY a JSON object in this exact format:
{"next_state": "state_name", "confidence": 0.8, "reasoning": "brief explanation", "suggested_actions": ["action1"], "response_strategy": "strategy_name", "cultural_considerations": {"key": "value"}}"""

        user_prompt = f"""Current state: {current_state}
User message: {user_message}
Recent conversation history:
{history_summary}

Analyze this conversation and provide your state management decision:"""

        return {
            "system": system_prompt,
            "user": user_prompt
        }

    def _parse_llm_state_decision(self, response: str) -> StateDecision:
        """Parse LLM response into structured decision"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                decision_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            # Validate and structure the decision
            return StateDecision(
                next_state=decision_data.get("next_state", ConversationState.AWAITING_ELABORATION.value),
                confidence=min(decision_data.get("confidence", 0.5), 1.0),
                reasoning=decision_data.get("reasoning", "LLM analysis"),
                suggested_actions=decision_data.get("suggested_actions", ["continue_conversation"]),
                response_strategy=decision_data.get("response_strategy", "information_gathering"),
                cultural_considerations=decision_data.get("cultural_considerations", {})
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to parse LLM decision: {e}")
            return self._fallback_state_decision(ConversationContext(
                session_id="", current_state=ConversationState.INITIAL_DISTRESS.value,
                conversation_history=[], user_message=""
            ))

    def _fallback_state_decision(self, context: ConversationContext) -> StateDecision:
        """Fallback state decision when LLM fails"""
        current_state = context.current_state
        user_message = context.user_message
        message_count = context.message_count

        # Progressive rule-based fallback that considers conversation history
        message_lower = user_message.lower()

        # Progressive rule-based fallback that considers conversation history
        message_lower = user_message.lower()

        # Check for questions first
        if "?" in user_message:
            next_state = ConversationState.AWAITING_CLARIFICATION.value
            strategy = "clarification_seeking"
        # Check for explicit help requests
        elif any(word in message_lower for word in ["help", "support", "advice", "what should", "how can"]):
            next_state = ConversationState.SUGGESTION_PENDING.value
            strategy = "solution_offering"
        # Progressive state advancement based on message count and current state
        elif current_state == ConversationState.INITIAL_DISTRESS.value:
            # First few messages - gather more information
            if message_count < 2:
                next_state = ConversationState.AWAITING_ELABORATION.value
                strategy = "information_gathering"
            else:
                next_state = ConversationState.SUGGESTION_PENDING.value
                strategy = "solution_offering"
        elif current_state == ConversationState.AWAITING_ELABORATION.value:
            # After elaboration, move to suggestions or clarification
            if message_count >= 3:
                next_state = ConversationState.SUGGESTION_PENDING.value
                strategy = "solution_offering"
            elif "?" in user_message:
                next_state = ConversationState.AWAITING_CLARIFICATION.value
                strategy = "clarification_seeking"
            else:
                next_state = ConversationState.AWAITING_ELABORATION.value
                strategy = "information_gathering"
        elif current_state == ConversationState.AWAITING_CLARIFICATION.value:
            # After clarification, provide guidance
            next_state = ConversationState.GUIDING_IN_PROGRESS.value
            strategy = "active_guidance"
        elif current_state == ConversationState.SUGGESTION_PENDING.value:
            # After suggestions, provide active guidance
            next_state = ConversationState.GUIDING_IN_PROGRESS.value
            strategy = "active_guidance"
        else:
            # Default to guiding progress for other states
            next_state = ConversationState.GUIDING_IN_PROGRESS.value
            strategy = "active_guidance"

        return StateDecision(
            next_state=next_state,
            confidence=0.6,  # Lower confidence for fallback
            reasoning=f"Fallback rule-based decision (message #{message_count})",
            suggested_actions=["continue_conversation"],
            response_strategy=strategy,
            cultural_considerations={}
        )

    async def generate_llm_response(self, context: ConversationContext, decision: StateDecision) -> str:
        """Generate LLM-powered response based on state and strategy"""
        strategy = decision.response_strategy

        # Build response prompt based on strategy
        response_prompt = self._build_response_prompt(context, decision)

        try:
            messages = [SystemMessage(content=response_prompt["system"]),
                       HumanMessage(content=response_prompt["user"])]

            # Check if LLM service is properly initialized
            if not self.llm_service or not self.llm_service.llm_provider:
                raise Exception("LLM service not properly initialized")

            response = await self.llm_service.llm_provider.generate_response(messages)
            return response

        except Exception as e:
            print(f"LLM response generation failed: {e}")
            return await self._get_fallback_response(context)

    def _build_response_prompt(self, context: ConversationContext, decision: StateDecision) -> Dict[str, str]:
        """Build prompt for LLM response generation"""
        current_state = context.current_state
        user_message = context.user_message
        strategy = decision.response_strategy
        cultural_considerations = decision.cultural_considerations

        # Strategy-specific instructions
        strategy_instructions = {
            "empathy_building": "Focus on validating feelings, building trust, and creating safety. Use warm, accepting language.",
            "information_gathering": "Ask gentle, open-ended questions to understand the user's situation better.",
            "clarification_seeking": "Ask specific questions to clarify details without overwhelming the user.",
            "solution_offering": "Offer practical, achievable coping strategies tailored to the user's situation.",
            "active_guidance": "Provide step-by-step guidance and support for implementing coping techniques.",
            "re_engagement": "Gently check in and invite continued sharing in a low-pressure way.",
            "crisis_protocol": "Provide immediate, supportive crisis intervention with resource information."
        }

        system_prompt = f"""You are a compassionate mental health support chatbot specializing in Rwandan cultural context.

Current conversation state: {current_state}
Response strategy: {strategy}
Instructions: {strategy_instructions.get(strategy, 'Provide supportive, culturally sensitive response')}

Guidelines:
- Be empathetic and non-judgmental
- Use culturally appropriate language and references
- Keep responses concise but comprehensive
- Focus on user's immediate needs
- Encourage help-seeking when appropriate
- Respect user's pace and boundaries"""

        user_prompt = f"""User message: {user_message}

Please provide a supportive response following the {strategy} strategy. Consider the Rwandan cultural context and current conversation state."""

        return {
            "system": system_prompt,
            "user": user_prompt
        }

    async def _get_fallback_response(self, context: ConversationContext) -> str:
        """Generate dynamic LLM response based on current state and user input"""
        current_state = context.current_state
        user_message = context.user_message
        strategy = context.emotion_data.get("strategy", "empathy_building") if context.emotion_data else "empathy_building"

        # Build context-aware prompt for LLM response generation
        system_prompt = self._build_dynamic_response_prompt(context, strategy)

        try:
            # Generate contextual response using LLM
            messages = [SystemMessage(content=system_prompt),
                       HumanMessage(content=user_message)]

            if not self.llm_service or not self.llm_service.llm_provider:
                # If LLM service not available, return a minimal contextual response
                return self._get_contextual_fallback(user_message, current_state)

            response = await self.llm_service.llm_provider.generate_response(messages)
            return response.strip()

        except Exception as e:
            print(f"Dynamic response generation failed: {e}")
            # Try one more time with a simpler approach
            try:
                # Fallback to simpler LLM prompt if available
                simple_prompt = f"""You are a supportive mental health chatbot.
Current state: {current_state}
User: {user_message}
Please provide a helpful, empathetic response."""

                messages = [SystemMessage(content=simple_prompt),
                           HumanMessage(content=user_message)]

                if self.llm_service and self.llm_service.llm_provider:
                    response = await self.llm_service.llm_provider.generate_response(messages)
                else:
                    raise Exception("LLM provider not available")
                return response.strip()
            except Exception as e2:
                print(f"Simple response generation also failed: {e2}")
                # Final fallback to contextual response
                return self._get_contextual_fallback(user_message, current_state)

    def _build_dynamic_response_prompt(self, context: ConversationContext, strategy: str) -> str:
        """Build dynamic prompt for LLM response generation"""
        current_state = context.current_state
        user_message = context.user_message
        query_validation = context.query_validation

        # Extract valuable information from query validation
        suggestions = query_validation.get("suggestions", [])
        emotion_data = query_validation.get("emotion_detection", {})
        detected_emotion = emotion_data.get("detected_emotion", "neutral")
        emotion_confidence = emotion_data.get("confidence", 0.0)
        emotion_intensity = emotion_data.get("intensity", "low")
        reasoning = query_validation.get("reasoning", "")
        keywords = query_validation.get("keywords_found", [])

        # State-specific context and instructions
        state_context = {
            ConversationState.INITIAL_DISTRESS.value: {
                "focus": "Show deep empathy and create emotional safety",
                "approach": "Validate feelings and build trust",
                "cultural_note": "Be warm and accepting in Rwandan cultural context"
            },
            ConversationState.AWAITING_ELABORATION.value: {
                "focus": "Gather more information gently",
                "approach": "Ask open-ended questions to understand better",
                "cultural_note": "Respect the person's pace of sharing"
            },
            ConversationState.AWAITING_CLARIFICATION.value: {
                "focus": "Clarify specific details",
                "approach": "Ask gentle, specific questions",
                "cultural_note": "Be patient and respectful"
            },
            ConversationState.SUGGESTION_PENDING.value: {
                "focus": "Offer practical coping strategies",
                "approach": "Provide culturally appropriate suggestions",
                "cultural_note": "Consider community and family support systems"
            },
            ConversationState.GUIDING_IN_PROGRESS.value: {
                "focus": "Provide step-by-step guidance",
                "approach": "Walk through techniques together",
                "cultural_note": "Use relatable, everyday examples"
            },
            ConversationState.CONVERSATION_IDLE.value: {
                "focus": "Gently re-engage",
                "approach": "Check in without pressure",
                "cultural_note": "Respect need for space"
            },
            ConversationState.CRISIS_INTERVENTION.value: {
                "focus": "Ensure immediate safety",
                "approach": "Connect to emergency resources",
                "cultural_note": "Mention local Rwandan helplines"
            }
        }

        context_info = state_context.get(current_state, state_context[ConversationState.INITIAL_DISTRESS.value])

        # Build suggestions text
        suggestions_text = ""
        if suggestions:
            suggestions_text = "\n".join(f"- {suggestion}" for suggestion in suggestions)

        # Build emotion context
        emotion_context = ""
        if detected_emotion and detected_emotion != "neutral":
            emotion_context = f"""
Detected Emotion: {detected_emotion} (confidence: {emotion_confidence:.2f}, intensity: {emotion_intensity})
Emotional Context: {reasoning}"""

        return f"""You are a compassionate mental health support chatbot specializing in Rwandan cultural context.

Current conversation state: {current_state}
Response strategy: {strategy}
Focus: {context_info['focus']}
Approach: {context_info['approach']}
Cultural consideration: {context_info['cultural_note']}

QUERY ANALYSIS CONTEXT:
{emotion_context}
Keywords identified: {', '.join(keywords) if keywords else 'None'}
Validation reasoning: {reasoning}

RECOMMENDED ACTIONS:
{suggestions_text}

Guidelines:
- Be empathetic and non-judgmental
- Use warm, supportive language
- Respect cultural values and communication styles
- Keep responses concise but comprehensive
- Focus on user's immediate emotional needs
- Encourage help-seeking when appropriate
- Incorporate the suggested actions above when relevant
- Consider the detected emotion in your response tone

User message: {user_message}

Please provide a supportive, culturally sensitive response that follows the recommended actions and considers the emotional context."""

    def _get_contextual_fallback(self, user_message: str, current_state: str) -> str:
        """Minimal contextual fallback when all else fails"""
        if current_state == ConversationState.CRISIS_INTERVENTION.value:
            return "I'm concerned about what you're going through. Please reach out to emergency services at 112 or the mental health helpline at 114."

        # Provide state-appropriate responses even when LLM is completely unavailable
        state_responses = {
            ConversationState.INITIAL_DISTRESS.value: "I can see you're going through a difficult time. I'm here to listen and support you through this.",
            ConversationState.AWAITING_ELABORATION.value: "Thank you for sharing that. Could you tell me more about what's been troubling you?",
            ConversationState.AWAITING_CLARIFICATION.value: "I want to make sure I understand you correctly. Could you help me clarify what you're experiencing?",
            ConversationState.SUGGESTION_PENDING.value: "Based on what you've shared, I think we can explore some helpful strategies together.",
            ConversationState.GUIDING_IN_PROGRESS.value: "Let's work through this together. What would be most helpful for you right now?",
            ConversationState.CONVERSATION_IDLE.value: "I'm here if you'd like to continue our conversation or explore this further.",
        }

        return state_responses.get(current_state, "I'm here to support you. How can I help you today?")

    async def route_conversation(self, session_id: str, user_message: str,
                                emotion_data: Optional[Dict[str, Any]] = None,
                                cultural_context: Optional[Dict[str, Any]] = None,
                                query_validation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main method to route conversation using LLM-enhanced decisions

        Returns enhanced response with LLM-powered state decisions
        """
        # Get current session
        session = self.session_manager.get_session(session_id)

        if not session:
            # Create new session with the provided session_id
            self.session_manager.create_session_with_id(
                session_id=session_id,
                user_id="user",
                initial_state=ConversationState.INITIAL_DISTRESS
            )
            session = self.session_manager.get_session(session_id)

        if not session:
            return {
                "response": "I'm here to support you. Could you tell me what's on your mind?",
                "next_state": ConversationState.INITIAL_DISTRESS.value,
                "response_type": "fallback"
            }

        # Prepare context
        context = ConversationContext(
            session_id=session_id,
            current_state=session.current_state.value,
            conversation_history=session.conversation_history,
            user_message=user_message,
            emotion_data=emotion_data,
            cultural_context=cultural_context,
            crisis_flags=session.crisis_flags,
            message_count=len(session.conversation_history),
            query_validation=query_validation
        )


        # Get LLM state decision with query validation context
        decision = await self.make_llm_state_decision(context)

        # Validate transition
        validated_state = self._validate_transition(session.current_state.value, decision.next_state)

        # Update session state
        next_state = ConversationState(validated_state)
        # Update session state
        self.session_manager.update_session_state(
            session_id,
            next_state,
            {
                "llm_decision": {
                    "next_state": decision.next_state,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning,
                    "strategy": decision.response_strategy
                },
                "last_llm_update": datetime.now().isoformat()
            }
        )

        # Generate LLM response
        response = await self.generate_llm_response(context, decision)

        return {
            "response": response,
            "next_state": validated_state,
            "response_type": decision.response_strategy,
            "confidence": decision.confidence,
            "llm_reasoning": decision.reasoning,
            "cultural_considerations": decision.cultural_considerations,
            "suggested_actions": decision.suggested_actions
        }

    def _validate_transition(self, current_state: str, proposed_state: str) -> str:
        """Validate the proposed state transition"""
        # More permissive validation for testing - allow more flexible transitions
        valid_transitions = {
            ConversationState.INITIAL_DISTRESS.value: [
                ConversationState.AWAITING_ELABORATION.value,
                ConversationState.AWAITING_CLARIFICATION.value,
                ConversationState.SUGGESTION_PENDING.value,
                ConversationState.CRISIS_INTERVENTION.value
            ],
            ConversationState.AWAITING_ELABORATION.value: [
                ConversationState.AWAITING_CLARIFICATION.value,
                ConversationState.SUGGESTION_PENDING.value,
                ConversationState.GUIDING_IN_PROGRESS.value,
                ConversationState.CRISIS_INTERVENTION.value
            ],
            ConversationState.AWAITING_CLARIFICATION.value: [
                ConversationState.GUIDING_IN_PROGRESS.value,
                ConversationState.AWAITING_ELABORATION.value,
                ConversationState.SUGGESTION_PENDING.value,
                ConversationState.CRISIS_INTERVENTION.value
            ],
            ConversationState.SUGGESTION_PENDING.value: [
                ConversationState.GUIDING_IN_PROGRESS.value,
                ConversationState.AWAITING_ELABORATION.value,
                ConversationState.CRISIS_INTERVENTION.value
            ],
            ConversationState.GUIDING_IN_PROGRESS.value: [
                ConversationState.CONVERSATION_IDLE.value,
                ConversationState.AWAITING_ELABORATION.value,
                ConversationState.CRISIS_INTERVENTION.value
            ],
            ConversationState.CONVERSATION_IDLE.value: [
                ConversationState.AWAITING_ELABORATION.value,
                ConversationState.INITIAL_DISTRESS.value,
                ConversationState.SUGGESTION_PENDING.value
            ],
            ConversationState.CRISIS_INTERVENTION.value: [
                ConversationState.AWAITING_ELABORATION.value,
                ConversationState.CRISIS_INTERVENTION.value,
                ConversationState.SUGGESTION_PENDING.value
            ]
        }

        valid_next_states = valid_transitions.get(current_state, [current_state])
        return proposed_state if proposed_state in valid_next_states else valid_next_states[0]


# Global LLM-enhanced state router instance (lazy initialization)
_llm_enhanced_router = None

def get_llm_enhanced_router():
    """Get or create the global LLM-enhanced state router with proper LLM service."""
    global _llm_enhanced_router
    if _llm_enhanced_router is None:
        try:
            # Try to get LLM service from service container
            from backend.app.services.service_container import get_service
            llm_service = get_service("llm_service")
            _llm_enhanced_router = LLMEnhancedStateRouter(llm_service=llm_service)
        except Exception:
            # Fallback to creating with default LLM service
            _llm_enhanced_router = LLMEnhancedStateRouter()
    return _llm_enhanced_router

# Backward compatibility - make llm_enhanced_router a property that returns the initialized instance
class _LLMEnhancedRouterProxy:
    """Proxy class to provide backward compatibility for llm_enhanced_router."""

    def __init__(self):
        self._router = None

    def _get_router(self):
        """Get the actual router instance."""
        if self._router is None:
            self._router = get_llm_enhanced_router()
        return self._router

    def __getattr__(self, name):
        """Delegate all attribute access to the actual router."""
        return getattr(self._get_router(), name)

# Create the proxy instance for backward compatibility
llm_enhanced_router = _LLMEnhancedRouterProxy()