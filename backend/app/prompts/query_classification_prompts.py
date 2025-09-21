"""
Query classification prompts for LangGraph workflow.

This module contains prompts specifically designed for classifying user queries
to determine if they are mental health related or random questions.
"""

from typing import Dict, List, Any
from enum import Enum


class QueryType(Enum):
    """Types of queries that can be identified"""
    MENTAL_SUPPORT = "mental_support"
    RANDOM_QUESTION = "random_question"
    UNCLEAR = "unclear"
    CRISIS = "crisis"


class QueryClassificationPrompts:
    """Prompts for classifying user queries using LLM."""

    @staticmethod
    def get_query_classification_prompt() -> str:
        """
        Get the prompt for classifying user queries.

        Returns:
            System prompt for query classification
        """
        return """You are an expert query classifier for a mental health support chatbot.

Your task is to analyze user queries and classify them into one of the following categories:

1. **MENTAL_SUPPORT** - Queries related to mental health, emotions, therapy, or emotional support
   Examples: "I'm feeling anxious", "I need help with depression", "Having panic attacks", "I'm stressed about work", "I feel overwhelmed"

2. **RANDOM_QUESTION** - General questions, casual conversation, technical questions, or non-mental-health topics
   Examples: "What's the weather?", "How do I install Python?", "What is the capital of France?", "How does photosynthesis work?", "What's 2+2?"

3. **UNCLEAR** - Queries that are too vague, empty, or need clarification
   Examples: "maybe", "xyz", empty messages, "idk", "not sure"

4. **CRISIS** - Queries indicating immediate danger or crisis situations
   Examples: "I want to kill myself", "I'm thinking about suicide", "I need emergency help", "I can't go on anymore"

CLASSIFICATION GUIDELINES:
- MENTAL_SUPPORT: Must contain emotional indicators, mental health terms, or clear intent to seek emotional support
- RANDOM_QUESTION: Technical questions, factual questions, programming questions, general knowledge questions, or casual conversation
- CRISIS: Only for immediate danger, suicidal ideation, self-harm, or emergency situations
- UNCLEAR: Only when genuinely unclear or empty

CRITICAL DISTINCTIONS:
- "What is Python?" = RANDOM_QUESTION (programming question)
- "What is depression?" = MENTAL_SUPPORT (mental health question)
- "How do I install Python?" = RANDOM_QUESTION (technical question)
- "How do I cope with anxiety?" = MENTAL_SUPPORT (mental health question)
- "I'm feeling like Python" = MENTAL_SUPPORT (emotional context)
- "Python is confusing" = RANDOM_QUESTION (technical complaint)

MENTAL HEALTH INDICATORS (must be present for MENTAL_SUPPORT):
- Emotions: anxious, depressed, sad, angry, stressed, overwhelmed, lonely, frustrated, worried, scared, hopeless
- Mental health terms: therapy, counseling, mental health, panic attack, trauma, grief, loss, heartbreak
- Support seeking: need help, feeling down, struggling with, having trouble, going through, dealing with
- Crisis: suicidal, self-harm, emergency, crisis (only for immediate danger)

RANDOM QUESTION INDICATORS:
- Programming: Python, JavaScript, Java, C++, programming, coding, development, software
- Technical: install, setup, configuration, bug, error, fix, troubleshoot, computer, software
- Factual: capital of, population of, weather, temperature, time, date, history, science
- General knowledge: what is, how does, when did, where is, who is, why do
- Casual: hey, hello, how are you, what's up, good morning, how's it going

RESPONSE FORMAT:
Return only a JSON object with this exact structure:
{
    "query_type": "MENTAL_SUPPORT|RANDOM_QUESTION|UNCLEAR|CRISIS",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of classification",
    "keywords_found": ["list", "of", "relevant", "keywords"],
    "is_crisis": true/false
}

CLASSIFICATION RULES:
1. If query contains programming/technical terms (Python, install, coding, etc.) → RANDOM_QUESTION
2. If query contains emotional words (feeling, anxious, stressed, etc.) → MENTAL_SUPPORT
3. If query is about mental health concepts → MENTAL_SUPPORT
4. If query is purely factual/technical → RANDOM_QUESTION
5. Only immediate danger → CRISIS
6. Default to RANDOM_QUESTION if unsure between MENTAL_SUPPORT and RANDOM_QUESTION

Be precise and consistent in your classification."""

    @staticmethod
    def get_crisis_detection_prompt() -> str:
        """
        Get the prompt for detecting crisis situations.

        Returns:
            System prompt for crisis detection
        """
        return """You are a crisis detection specialist for mental health emergencies.

Your task is to analyze user queries and determine if they indicate an immediate crisis situation.

CRISIS INDICATORS:
- Suicidal ideation: "kill myself", "end my life", "suicide", "want to die"
- Self-harm: "hurt myself", "cut myself", "self-harm", "harm myself"
- Immediate danger: "emergency", "crisis", "help me now", "can't go on"
- Severe distress: "end it all", "life not worth living", "better off dead"

NON-CRISIS (but still serious):
- General sadness: "feeling sad", "depressed", "anxious"
- Past experiences: "I used to", "last week", "before"
- Hypothetical: "what if", "sometimes I think"

RESPONSE FORMAT:
Return only a JSON object with this exact structure:
{
    "is_crisis": true/false,
    "confidence": 0.0-1.0,
    "crisis_indicators": ["list", "of", "detected", "indicators"],
    "severity": "low|medium|high|critical",
    "recommended_action": "monitor|immediate_intervention|professional_referral"
}

CRISIS LEVELS:
- low: General mental health concerns, sadness, anxiety
- medium: Self-harm thoughts, depression, but not immediate
- high: Specific self-harm plans, suicidal ideation
- critical: Immediate danger, active crisis, emergency language"""

    @staticmethod
    def get_query_suggestions_prompt() -> str:
        """
        Get the prompt for generating query handling suggestions.

        Returns:
            System prompt for query suggestions
        """
        return """You are a query routing specialist for a mental health chatbot.

Based on the query classification, provide specific suggestions for how to handle the user's query.

RESPONSE FORMAT:
Return only a JSON object with this exact structure:
{
    "suggestions": [
        "Specific suggestion 1",
        "Specific suggestion 2",
        "Specific suggestion 3"
    ],
    "routing_priority": "high|medium|low",
    "requires_human_intervention": true/false,
    "follow_up_questions": ["question1", "question2"]
}

SUGGESTION GUIDELINES:
- For MENTAL_SUPPORT: Focus on empathy, validation, and support
- For RANDOM_QUESTION: Provide information, keep it light
- For UNCLEAR: Ask for clarification, provide examples
- For CRISIS: Immediate professional referral, safety first"""

    @staticmethod
    def get_emotion_detection_prompt() -> str:
        """
        Get the prompt for detecting emotions in user queries.

        Returns:
            System prompt for emotion detection
        """
        return """You are an emotion detection specialist for mental health support.

Your task is to analyze user queries and detect the emotional state and intensity of the user.

EMOTIONS TO DETECT:
- joy, sadness, anger, fear, surprise, disgust, neutral, anxiety
- Also detect: stress, overwhelmed, frustrated, lonely, hopeless, worried, scared

EMOTION INTENSITY LEVELS:
- low: Mild emotional indicators
- medium: Moderate emotional expression
- high: Strong emotional language
- very_high: Intense emotional expression

CONTEXT RELEVANCE:
- very_low: Emotion not relevant to mental health context
- low: Mild relevance to mental health
- medium: Moderate relevance to mental health
- high: Strong relevance to mental health
- very_high: Critical relevance to mental health

RESPONSE FORMAT:
Return only a JSON object with this exact structure:
{
    "detected_emotion": "primary_emotion",
    "confidence": 0.0-1.0,
    "emotion_scores": {
        "joy": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "neutral": 1.0,
        "anxiety": 0.0
    },
    "reasoning": "Explanation of detected emotions",
    "intensity": "low|medium|high|very_high",
    "context_relevance": "very_low|low|medium|high|very_high"
}

ANALYSIS GUIDELINES:
- Consider the overall tone and emotional language used
- Look for emotional words and their intensity modifiers
- Assess how relevant the emotions are to mental health context
- Provide confidence score based on clarity of emotional expression
- Default to "neutral" if no clear emotions detected"""

    @staticmethod
    def parse_classification_response(response: str) -> Dict[str, Any]:
        """
        Parse the LLM response for query classification.

        Args:
            response: Raw response from the LLM

        Returns:
            Parsed classification result
        """
        import json
        import re

        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback parsing if JSON extraction fails
        return {
            "query_type": "UNCLEAR",
            "confidence": 0.5,
            "reasoning": "Failed to parse classification response",
            "keywords_found": [],
            "is_crisis": False
        }