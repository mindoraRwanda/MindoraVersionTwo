"""
Query Validation Service
Validates and categorizes user queries to determine if they are mental support related or random questions.
"""

import re
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class QueryType(Enum):
    """Types of queries that can be identified"""
    MENTAL_SUPPORT = "mental_support"
    RANDOM_QUESTION = "random_question"
    UNCLEAR = "unclear"


@dataclass
class ValidationResult:
    """Result of query validation"""
    query_type: QueryType
    confidence: float
    keywords_found: List[str]
    reasoning: str
    timestamp: datetime
    needs_clarification: bool = False


class QueryValidatorService:
    """
    Service for validating and categorizing user queries
    """

    def __init__(self):
        """Initialize the query validator with keyword sets"""
        self._initialize_keyword_sets()

    def _initialize_keyword_sets(self):
        """Initialize keyword sets for different query types"""

        # Mental health and emotional support keywords
        self.mental_health_keywords = {
            # Emotions and feelings
            'anxious', 'anxiety', 'worried', 'worry', 'stressed', 'stress',
            'depressed', 'depression', 'sad', 'sadness', 'angry', 'anger',
            'frustrated', 'frustration', 'overwhelmed', 'lonely', 'loneliness',
            'hopeless', 'helpless', 'worthless', 'guilty', 'shame', 'fear',
            'scared', 'panic', 'trauma', 'grief', 'loss', 'heartbroken',

            # Mental health conditions
            'mental health', 'therapy', 'counseling', 'therapist', 'psychologist',
            'psychiatrist', 'counselor', 'panic attack', 'ptsd', 'bipolar',
            'schizophrenia', 'ocd', 'eating disorder', 'addiction', 'substance abuse',

            # Support seeking phrases
            'need help', 'need support', 'feeling down', 'going through',
            'struggling with', 'having trouble', 'feeling lost', 'need advice',
            'talk to someone', 'feeling overwhelmed', 'having a hard time',

            # Crisis indicators
            'suicidal', 'suicide', 'self harm', 'hurting myself', 'end it all',
            'want to die', 'crisis', 'emergency', 'help me', 'save me'
        }

        # Random question indicators
        self.random_question_indicators = {
            # General knowledge questions
            'what is', 'how does', 'explain', 'define', 'tell me about',
            'how to', 'what are', 'when did', 'where is', 'who is',

            # Casual conversation
            'how are you', 'what\'s up', 'hey', 'hello', 'hi there',
            'good morning', 'good afternoon', 'good evening', 'how\'s it going',

            # Weather and time
            'weather', 'temperature', 'time is it', 'what time', 'forecast',

            # Technical questions
                'how to use', 'bug', 'error', 'problem with',
                'not working', 'fix', 'troubleshoot',

            # Factual questions
            'capital of', 'population of', 'how many', 'how much', 'what year'
        }

        # Context words that might indicate mental health even with general questions
        self.context_indicators = {
            'feel', 'feeling', 'felt', 'emotion', 'emotional', 'mentally',
            'psychologically', 'mood', 'state of mind', 'headspace'
        }

    def validate_query(self, query: str) -> ValidationResult:
        """
        Validate and categorize a user query

        Args:
            query: The user's query string

        Returns:
            ValidationResult with categorization and confidence
        """
        if not query or not query.strip():
            return ValidationResult(
                query_type=QueryType.UNCLEAR,
                confidence=1.0,
                keywords_found=[],
                reasoning="Empty or invalid query",
                timestamp=datetime.now(),
                needs_clarification=True
            )

        query_lower = query.lower().strip()
        keywords_found = []

        # Check for mental health keywords
        mental_health_score = self._calculate_mental_health_score(query_lower, keywords_found)

        # Check for random question indicators
        random_question_score = self._calculate_random_question_score(query_lower, keywords_found)

        # Determine query type based on scores
        if mental_health_score > random_question_score and mental_health_score > 0.3:
            query_type = QueryType.MENTAL_SUPPORT
            confidence = min(mental_health_score, 1.0)
            reasoning = f"Mental health indicators detected with score {mental_health_score:.2f}"
        elif random_question_score > mental_health_score and random_question_score > 0.2:
            query_type = QueryType.RANDOM_QUESTION
            confidence = min(random_question_score, 1.0)
            reasoning = f"Random question indicators detected with score {random_question_score:.2f}"
        else:
            query_type = QueryType.UNCLEAR
            confidence = 0.5
            reasoning = "Query type unclear - may need clarification"
            needs_clarification = True

        return ValidationResult(
            query_type=query_type,
            confidence=confidence,
            keywords_found=keywords_found,
            reasoning=reasoning,
            timestamp=datetime.now(),
            needs_clarification=query_type == QueryType.UNCLEAR
        )

    def _calculate_mental_health_score(self, query: str, keywords_found: List[str]) -> float:
        """Calculate score for mental health relevance"""
        score = 0.0
        words = query.split()

        # Check for exact mental health keywords
        for keyword in self.mental_health_keywords:
            if keyword in query:
                score += 0.4
                keywords_found.append(keyword)

        # Check for partial matches and context
        for word in words:
            # Remove punctuation for better matching
            clean_word = re.sub(r'[^\w]', '', word)

            # Check for emotional context words
            if clean_word in self.context_indicators:
                score += 0.2

            # Check for partial matches of mental health terms
            for mh_keyword in self.mental_health_keywords:
                if mh_keyword in clean_word or clean_word in mh_keyword:
                    score += 0.1

        # Boost score for crisis indicators
        crisis_words = ['suicidal', 'suicide', 'self harm', 'crisis', 'emergency']
        for crisis_word in crisis_words:
            if crisis_word in query:
                score += 0.5
                break

        return min(score, 1.0)

    def _calculate_random_question_score(self, query: str, keywords_found: List[str]) -> float:
        """Calculate score for random question relevance"""
        score = 0.0
        words = query.split()

        # Check for question patterns
        if query.startswith(('what', 'how', 'when', 'where', 'why', 'who', 'which')):
            score += 0.3

        if '?' in query:
            score += 0.2

        # Check for random question indicators
        for indicator in self.random_question_indicators:
            if indicator in query:
                score += 0.3
                keywords_found.append(indicator)

        # Check for casual conversation patterns
        casual_patterns = [
            r'^hey|^hi|^hello',
            r'how are you|what\'s up|how\'s it going',
            r'good (morning|afternoon|evening)'
        ]

        for pattern in casual_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.2

        return min(score, 1.0)

    def get_query_suggestions(self, result: ValidationResult) -> List[str]:
        """
        Get suggestions for handling the query based on validation result

        Args:
            result: ValidationResult from query validation

        Returns:
            List of suggested actions or responses
        """
        suggestions = []

        if result.query_type == QueryType.MENTAL_SUPPORT:
            suggestions.extend([
                "Route to mental health support system",
                "Apply emotional context analysis",
                "Use therapeutic response patterns",
                "Monitor for crisis indicators"
            ])
        elif result.query_type == QueryType.RANDOM_QUESTION:
            suggestions.extend([
                "Provide factual information",
                "Use general knowledge base",
                "Keep response light and conversational",
                "Offer to redirect to mental health support if needed"
            ])
        else:
            suggestions.extend([
                "Ask for clarification",
                "Provide examples of supported query types",
                "Offer both mental health and general assistance options"
            ])

        return suggestions

    def is_crisis_query(self, query: str) -> Tuple[bool, float]:
        """
        Check if query indicates a crisis situation

        Args:
            query: The user's query string

        Returns:
            Tuple of (is_crisis, confidence_score)
        """
        query_lower = query.lower()

        crisis_indicators = [
            'suicidal', 'suicide', 'kill myself', 'end it all', 'want to die',
            'self harm', 'hurting myself', 'crisis', 'emergency', 'help me',
            'save me', 'can\'t go on', 'life not worth living'
        ]

        crisis_score = 0.0
        found_indicators = []

        for indicator in crisis_indicators:
            if indicator in query_lower:
                crisis_score += 0.8
                found_indicators.append(indicator)

        # Additional context checks
        if any(word in query_lower for word in ['now', 'today', 'right now', 'immediately']):
            crisis_score += 0.2

        return crisis_score > 0.6, min(crisis_score, 1.0)


def initialize_query_validator() -> QueryValidatorService:
    """Initialize and return a query validator service instance"""
    return QueryValidatorService()