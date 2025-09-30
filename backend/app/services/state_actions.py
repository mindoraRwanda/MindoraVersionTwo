"""
State Actions for Conversational Finite State Machine

This module implements specific actions and response generation for each
conversation state in the mental health chatbot system.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from backend.app.services.session_state_manager import (
    SessionStateManager, ConversationState, session_manager
)
from backend.app.services.crisis_interceptor import crisis_interceptor
from backend.app.services.conversation_content import ConversationContentManager
from backend.app.services.state_router import state_router


class StateActions:
    """Handles actions and responses for each conversation state"""

    def __init__(self):
        self.content_manager = ConversationContentManager()

    async def execute_state_action(self, session_id: str, user_message: str,
                                 emotion_data: Optional[Dict[str, Any]] = None,
                                 cultural_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the appropriate action for the current conversation state

        Returns response data including message, state transition, and metadata
        """
        session = session_manager.get_session(session_id)
        if not session:
            # Create new session
            session_id = session_manager.create_session("user")
            session = session_manager.get_session(session_id)

        if session is None:
            # Fallback if session creation failed
            return {
                "response": "I'm here to support you. Could you tell me what's been going on?",
                "current_state": ConversationState.INITIAL_DISTRESS.value,
                "suggested_actions": ["start_conversation"],
                "response_type": "fallback"
            }

        current_state = session.current_state

        # Check for crisis override
        if crisis_interceptor.should_override_conversation(session_id):
            return await self._handle_crisis_action(session_id, user_message, emotion_data)

        # Route to appropriate state handler
        state_handlers = {
            ConversationState.INITIAL_DISTRESS: self._handle_initial_distress,
            ConversationState.AWAITING_ELABORATION: self._handle_awaiting_elaboration,
            ConversationState.AWAITING_CLARIFICATION: self._handle_awaiting_clarification,
            ConversationState.SUGGESTION_PENDING: self._handle_suggestion_pending,
            ConversationState.GUIDING_IN_PROGRESS: self._handle_guiding_in_progress,
            ConversationState.CONVERSATION_IDLE: self._handle_conversation_idle,
            ConversationState.CRISIS_INTERVENTION: self._handle_crisis_action,
        }

        handler = state_handlers.get(current_state, self._handle_initial_distress)
        response_data = await handler(session_id, user_message, emotion_data, cultural_context)

        # Determine next state
        next_state = state_router.determine_next_state(session_id, user_message, emotion_data)
        response_data['next_state'] = next_state.value

        # Update session state if changed
        if next_state != current_state:
            session_manager.update_session_state(session_id, next_state)

        return response_data

    async def _handle_initial_distress(self, session_id: str, user_message: str,
                                    emotion_data: Optional[Dict[str, Any]] = None,
                                    cultural_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle initial distress state - build empathy and trust"""
        session = session_manager.get_session(session_id)

        # Get empathy-focused content
        empathy_content = self.content_manager.get_content("empathy", "initial_response")

        # Personalize based on emotion data
        emotion_context = self._analyze_emotion_context(emotion_data)

        response = self._build_empathy_response(user_message, emotion_context, cultural_context)

        # Add follow-up question to encourage elaboration
        follow_up = self.content_manager.get_content("questions", "elaboration_prompt")

        full_response = f"{response}\n\n{follow_up}"

        # Add to session history
        session_manager.add_message_to_history(session_id, "user", user_message)
        session_manager.add_message_to_history(session_id, "assistant", full_response)

        return {
            "response": full_response,
            "current_state": ConversationState.INITIAL_DISTRESS.value,
            "emotion_context": emotion_context,
            "suggested_actions": ["acknowledge_feelings", "build_trust", "encourage_sharing"],
            "response_type": "empathy_building"
        }

    async def _handle_awaiting_elaboration(self, session_id: str, user_message: str,
                                        emotion_data: Optional[Dict[str, Any]] = None,
                                        cultural_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle awaiting elaboration state - gather more details"""
        session = session_manager.get_session(session_id)

        # Analyze what details the user provided
        detail_analysis = self._analyze_user_details(user_message)

        # Get appropriate response based on detail level
        if detail_analysis["detail_level"] == "low":
            response = self.content_manager.get_content("clarification", "need_more_details")
        elif detail_analysis["detail_level"] == "medium":
            response = self.content_manager.get_content("validation", "partial_understanding")
        else:
            response = self.content_manager.get_content("empathy", "deep_understanding")

        # Add cultural context if relevant
        if cultural_context:
            response = self._integrate_cultural_context(response, cultural_context)

        # Suggest next steps
        next_steps = self._suggest_next_steps(detail_analysis, emotion_data)

        full_response = f"{response}\n\n{next_steps}"

        # Update session history
        session_manager.add_message_to_history(session_id, "user", user_message)
        session_manager.add_message_to_history(session_id, "assistant", full_response)

        return {
            "response": full_response,
            "current_state": ConversationState.AWAITING_ELABORATION.value,
            "detail_analysis": detail_analysis,
            "suggested_actions": ["gather_details", "validate_understanding", "provide_reassurance"],
            "response_type": "information_gathering"
        }

    async def _handle_awaiting_clarification(self, session_id: str, user_message: str,
                                          emotion_data: Optional[Dict[str, Any]] = None,
                                          cultural_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle clarification state - seek specific understanding"""
        session = session_manager.get_session(session_id)

        # Identify what needs clarification
        clarification_needs = self._identify_clarification_needs(user_message, session)

        # Get clarification response
        if clarification_needs["type"] == "timeline":
            response = self.content_manager.get_content("clarification", "timeline_question")
        elif clarification_needs["type"] == "intensity":
            response = self.content_manager.get_content("clarification", "intensity_question")
        elif clarification_needs["type"] == "context":
            response = self.content_manager.get_content("clarification", "context_question")
        else:
            response = self.content_manager.get_content("clarification", "general_question")

        # Add empathetic framing
        empathetic_framing = self.content_manager.get_content("empathy", "clarification_framing")
        full_response = f"{empathetic_framing}\n\n{response}"

        # Update session history
        session_manager.add_message_to_history(session_id, "user", user_message)
        session_manager.add_message_to_history(session_id, "assistant", full_response)

        return {
            "response": full_response,
            "current_state": ConversationState.AWAITING_CLARIFICATION.value,
            "clarification_needs": clarification_needs,
            "suggested_actions": ["ask_specific_questions", "provide_examples", "normalize_questions"],
            "response_type": "clarification_seeking"
        }

    async def _handle_suggestion_pending(self, session_id: str, user_message: str,
                                      emotion_data: Optional[Dict[str, Any]] = None,
                                      cultural_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle suggestion pending state - offer coping strategies"""
        session = session_manager.get_session(session_id)

        # Get conversation context for personalized suggestions
        context_summary = self._summarize_conversation_context(session)

        # Generate appropriate suggestions based on context
        suggestions = self._generate_personalized_suggestions(context_summary, emotion_data, cultural_context)

        # Format suggestions in culturally appropriate way
        formatted_suggestions = self._format_suggestions_culturally(suggestions, cultural_context)

        # Add introduction and conclusion
        intro = self.content_manager.get_content("suggestions", "introduction")
        conclusion = self.content_manager.get_content("suggestions", "conclusion")

        full_response = f"{intro}\n\n{formatted_suggestions}\n\n{conclusion}"

        # Update session history
        session_manager.add_message_to_history(session_id, "user", user_message)
        session_manager.add_message_to_history(session_id, "assistant", full_response)

        return {
            "response": full_response,
            "current_state": ConversationState.SUGGESTION_PENDING.value,
            "suggestions": suggestions,
            "suggested_actions": ["offer_coping_strategies", "provide_resources", "encourage_practice"],
            "response_type": "solution_offering"
        }

    async def _handle_guiding_in_progress(self, session_id: str, user_message: str,
                                        emotion_data: Optional[Dict[str, Any]] = None,
                                        cultural_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle active guidance state - provide step-by-step support"""
        session = session_manager.get_session(session_id)

        # Assess progress and determine guidance type
        guidance_type = self._determine_guidance_type(user_message, session)

        # Get appropriate guidance content
        if guidance_type == "step_by_step":
            response = self._get_step_by_step_guidance(user_message, cultural_context)
        elif guidance_type == "technique_practice":
            response = self._get_technique_practice(user_message, cultural_context)
        elif guidance_type == "progress_check":
            response = self._get_progress_check_response(user_message, session)
        else:
            response = self._get_general_guidance(user_message, cultural_context)

        # Add encouragement and next steps
        encouragement = self.content_manager.get_content("encouragement", "ongoing_support")
        full_response = f"{response}\n\n{encouragement}"

        # Update session history
        session_manager.add_message_to_history(session_id, "user", user_message)
        session_manager.add_message_to_history(session_id, "assistant", full_response)

        return {
            "response": full_response,
            "current_state": ConversationState.GUIDING_IN_PROGRESS.value,
            "guidance_type": guidance_type,
            "suggested_actions": ["provide_structure", "offer_practice", "monitor_progress"],
            "response_type": "active_guidance"
        }

    async def _handle_conversation_idle(self, session_id: str, user_message: str,
                                      emotion_data: Optional[Dict[str, Any]] = None,
                                      cultural_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle idle conversation state - re-engage user"""
        session = session_manager.get_session(session_id)

        # Check if user is re-engaging or still disengaged
        engagement_level = self._assess_engagement_level(user_message)

        if engagement_level == "high":
            response = self.content_manager.get_content("re_engagement", "welcome_back")
        elif engagement_level == "medium":
            response = self.content_manager.get_content("re_engagement", "gentle_check_in")
        else:
            response = self.content_manager.get_content("re_engagement", "minimal_response")

        # Update session history
        session_manager.add_message_to_history(session_id, "user", user_message)
        session_manager.add_message_to_history(session_id, "assistant", response)

        return {
            "response": response,
            "current_state": ConversationState.CONVERSATION_IDLE.value,
            "engagement_level": engagement_level,
            "suggested_actions": ["check_in_gently", "offer_continued_support", "respect_boundaries"],
            "response_type": "re_engagement"
        }

    async def _handle_crisis_action(self, session_id: str, user_message: str,
                                  emotion_data: Optional[Dict[str, Any]] = None,
                                  cultural_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle crisis intervention state"""
        # Get crisis response from interceptor
        crisis_response = await crisis_interceptor.intercept_and_respond(
            session_id, user_message, emotion_data, "rwanda"
        )

        if crisis_response:
            # Update session history
            session_manager.add_message_to_history(
                session_id, "user", user_message,
                {"crisis_indicators": crisis_response.get("crisis_flags", [])}
            )
            session_manager.add_message_to_history(
                session_id, "assistant", crisis_response["immediate_response"],
                {"crisis_intervention": True}
            )

            return {
                "response": crisis_response["immediate_response"],
                "current_state": ConversationState.CRISIS_INTERVENTION.value,
                "crisis_data": crisis_response,
                "suggested_actions": ["provide_resources", "encourage_help_seeking", "follow_up"],
                "response_type": "crisis_intervention"
            }

        # Fallback if crisis detection failed
        return await self._handle_initial_distress(session_id, user_message, emotion_data, cultural_context)

    # Helper methods for response generation

    def _analyze_emotion_context(self, emotion_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze emotion data for response personalization"""
        if not emotion_data:
            return {"primary_emotion": "neutral", "intensity": "low"}

        # Find primary emotion
        primary_emotion = "neutral"
        max_intensity = 0

        for emotion, intensity in emotion_data.items():
            if isinstance(intensity, (int, float)) and intensity > max_intensity:
                max_intensity = intensity
                primary_emotion = emotion

        return {
            "primary_emotion": primary_emotion,
            "intensity": "high" if max_intensity > 0.7 else "medium" if max_intensity > 0.4 else "low",
            "all_emotions": emotion_data
        }

    def _build_empathy_response(self, user_message: str, emotion_context: Dict[str, Any],
                              cultural_context: Optional[Dict[str, Any]] = None) -> str:
        """Build empathetic response based on user message and emotions"""
        # Get base empathy content
        if emotion_context["intensity"] == "high":
            base_response = self.content_manager.get_content("empathy", "high_intensity")
        elif emotion_context["intensity"] == "medium":
            base_response = self.content_manager.get_content("empathy", "medium_intensity")
        else:
            base_response = self.content_manager.get_content("empathy", "low_intensity")

        # Personalize based on primary emotion
        emotion_templates = {
            "sadness": "I can hear the sadness in your words, and I'm here to support you through this.",
            "fear": "I can sense your fear, and I want you to know that it's okay to feel scared.",
            "anger": "I can feel your anger, and it's completely valid to feel this way.",
            "joy": "I'm glad to hear some joy in your voice, even amidst the challenges.",
            "neutral": "I hear what you're going through, and I want to understand more."
        }

        emotion_intro = emotion_templates.get(emotion_context["primary_emotion"],
                                           emotion_templates["neutral"])

        return f"{emotion_intro}\n\n{base_response}"

    def _analyze_user_details(self, message: str) -> Dict[str, Any]:
        """Analyze level of detail in user message"""
        word_count = len(message.split())

        # Check for specific indicators
        has_timeline = any(word in message.lower() for word in ["when", "before", "after", "during", "recently"])
        has_intensity = any(word in message.lower() for word in ["very", "extremely", "really", "so", "intense"])
        has_context = any(word in message.lower() for word in ["because", "situation", "circumstances", "context"])

        if word_count < 10:
            detail_level = "low"
        elif word_count < 30:
            detail_level = "medium"
        else:
            detail_level = "high"

        return {
            "detail_level": detail_level,
            "word_count": word_count,
            "has_timeline": has_timeline,
            "has_intensity": has_intensity,
            "has_context": has_context
        }

    def _integrate_cultural_context(self, response: str, cultural_context: Dict[str, Any]) -> str:
        """Integrate cultural context into response"""
        if not cultural_context:
            return response

        # Add cultural sensitivity based on context
        cultural_tone = cultural_context.get("tone", "respectful")
        cultural_values = cultural_context.get("values", [])

        if "community" in cultural_values:
            response += "\n\nRemember that in your community, seeking support shows strength, not weakness."
        if "family" in cultural_values:
            response += "\n\nYour family and loved ones care about you and want to support you."

        return response

    def _suggest_next_steps(self, detail_analysis: Dict[str, Any],
                          emotion_data: Optional[Dict[str, Any]] = None) -> str:
        """Suggest appropriate next steps based on analysis"""
        if detail_analysis["detail_level"] == "low":
            return "Could you tell me more about what's been going on? I'm here to listen."
        elif detail_analysis["detail_level"] == "medium":
            return "Thank you for sharing that. Would you like to tell me more about how this has been affecting you?"
        else:
            return "I appreciate you sharing these details with me. Let's work together to find some ways to help you through this."

    def _identify_clarification_needs(self, message: str, session: Any) -> Dict[str, Any]:
        """Identify what aspects need clarification"""
        message_lower = message.lower()

        clarification_type = "general"
        if any(word in message_lower for word in ["when", "how long", "how often"]):
            clarification_type = "timeline"
        elif any(word in message_lower for word in ["how much", "how bad", "intensity"]):
            clarification_type = "intensity"
        elif any(word in message_lower for word in ["where", "who", "what happened"]):
            clarification_type = "context"

        return {
            "type": clarification_type,
            "confidence": "high" if len(message.split()) > 15 else "medium"
        }

    def _summarize_conversation_context(self, session: Any) -> Dict[str, Any]:
        """Summarize conversation context for personalized suggestions"""
        if not session.conversation_history:
            return {"themes": [], "intensity": "unknown", "duration": "unknown"}

        # Extract themes from conversation
        themes = []
        user_messages = [msg for msg in session.conversation_history if msg.get("role") == "user"]

        all_text = " ".join([msg.get("content", "") for msg in user_messages])
        text_lower = all_text.lower()

        # Identify common themes
        theme_keywords = {
            "anxiety": ["anxious", "worried", "nervous", "panic", "fear"],
            "depression": ["sad", "depressed", "hopeless", "empty", "tired"],
            "stress": ["stressed", "overwhelmed", "pressure", "too much"],
            "relationship": ["relationship", "partner", "family", "friend", "alone"],
            "work": ["work", "job", "boss", "colleague", "career"]
        }

        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme)

        return {
            "themes": themes,
            "message_count": len(user_messages),
            "conversation_age": (datetime.now() - session.created_at).total_seconds() / 60  # minutes
        }

    def _generate_personalized_suggestions(self, context: Dict[str, Any],
                                        emotion_data: Optional[Dict[str, Any]] = None,
                                        cultural_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate personalized suggestions based on context"""
        suggestions = []

        # Base suggestions for common themes
        if "anxiety" in context["themes"]:
            suggestions.extend([
                "Practice deep breathing exercises when you feel anxious",
                "Try grounding techniques, like focusing on your senses",
                "Consider talking to someone you trust about your worries"
            ])
        if "depression" in context["themes"]:
            suggestions.extend([
                "Try small, achievable activities each day",
                "Connect with supportive people in your life",
                "Consider gentle physical activity, like walking"
            ])
        if "stress" in context["themes"]:
            suggestions.extend([
                "Break down overwhelming tasks into smaller steps",
                "Practice relaxation techniques regularly",
                "Ensure you're getting enough rest"
            ])

        # Add general suggestions if no specific themes
        if not suggestions:
            suggestions.extend([
                "Take care of your basic needs - sleep, nutrition, and gentle movement",
                "Connect with supportive people in your life",
                "Practice self-compassion and be kind to yourself"
            ])

        return suggestions[:3]  # Limit to 3 suggestions

    def _format_suggestions_culturally(self, suggestions: List[str],
                                     cultural_context: Optional[Dict[str, Any]] = None) -> str:
        """Format suggestions in culturally appropriate way"""
        if not suggestions:
            return "I'm here to support you in ways that feel right for you."

        formatted = "Here are some suggestions that might help:\n\n"
        for i, suggestion in enumerate(suggestions, 1):
            formatted += f"{i}. {suggestion}\n"

        # Add cultural sensitivity
        if cultural_context and "community" in cultural_context.get("values", []):
            formatted += "\nRemember, seeking help is a sign of strength in your community."

        return formatted

    def _determine_guidance_type(self, message: str, session: Any) -> str:
        """Determine type of guidance to provide"""
        message_lower = message.lower()

        if any(word in message_lower for word in ["how", "steps", "guide"]):
            return "step_by_step"
        elif any(word in message_lower for word in ["practice", "try", "exercise"]):
            return "technique_practice"
        elif any(word in message_lower for word in ["better", "improving", "progress"]):
            return "progress_check"
        else:
            return "general_guidance"

    def _get_step_by_step_guidance(self, message: str, cultural_context: Optional[Dict[str, Any]] = None) -> str:
        """Get step-by-step guidance response"""
        return self.content_manager.get_content("guidance", "step_by_step")

    def _get_technique_practice(self, message: str, cultural_context: Optional[Dict[str, Any]] = None) -> str:
        """Get technique practice response"""
        return self.content_manager.get_content("guidance", "technique_practice")

    def _get_progress_check_response(self, message: str, session: Any) -> str:
        """Get progress check response"""
        return self.content_manager.get_content("guidance", "progress_check")

    def _get_general_guidance(self, message: str, cultural_context: Optional[Dict[str, Any]] = None) -> str:
        """Get general guidance response"""
        return self.content_manager.get_content("guidance", "general_support")

    def _assess_engagement_level(self, message: str) -> str:
        """Assess user's engagement level"""
        if not message or len(message.strip()) < 3:
            return "low"
        elif len(message.split()) < 10:
            return "medium"
        else:
            return "high"


# Global state actions instance
state_actions = StateActions()