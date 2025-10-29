import json
import logging
from typing import Dict, Any, Optional, List
from backend.app.services.llm_providers import LLMProvider
from backend.app.prompts.cultural_context_prompts import CulturalContextPrompts

logger = logging.getLogger(__name__)

class LLMEmotionClassifier:
    """LLM-powered emotion classifier with cultural context integration."""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize the LLM emotion classifier."""
        self.llm_provider = llm_provider
        self.cultural_prompts = CulturalContextPrompts()
        # Language detection removed - default to English
        logger.info("🧠 LLM Emotion Classifier initialized")
    
    def _detect_language(self, text: str) -> str:
        """Detect language from text."""
        # Language detection removed - default to English
        return "en"
    
    def _get_cultural_context(self, language: str) -> Dict[str, Any]:
        """Get cultural context for emotion detection."""
        return {
            "language": language,
            "cultural_context": self.cultural_prompts.get_rwanda_cultural_context(language),
            "emotion_responses": self.cultural_prompts.get_emotion_responses(language)
        }
    
    def _get_cultural_integration_prompt(self, language: str, gender: Optional[str] = None) -> str:
        """Get cultural integration prompt for emotion detection."""
        return self.cultural_prompts.get_cultural_integration_prompt(language, gender)
    
    async def classify_emotion(
        self, 
        text: str, 
        user_gender: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify emotion using LLM analysis with cultural context.
        
        Args:
            text: Input text to analyze
            user_gender: User's gender for cultural addressing
            context: Additional context for analysis
            
        Returns:
            Dictionary with emotion classification results
        """
        if not self.llm_provider:
            logger.warning("No LLM provider available, returning neutral emotion")
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "reasoning": "No LLM provider available",
                "keywords": [],
                "cultural_context": {}
            }
        
        try:
            # Detect language
            language = self._detect_language(text)
            logger.info(f"🌍 Detected language: {language}")
            
            # Get cultural context
            cultural_context = self._get_cultural_context(language)
            cultural_prompt = self._get_cultural_integration_prompt(language, user_gender)
            
            # Build system prompt for emotion detection
            system_prompt = f"""
            You are an expert emotion detection specialist for youth mental health with cultural awareness for {language} speakers.
            
            {cultural_prompt}
            
            Analyze the text for emotions with cultural sensitivity:
            1. Primary emotion and confidence score (0-1)
            2. List of emotion keywords (including cultural expressions)
            3. Detailed reasoning for emotion detection
            4. Emotion intensity level (low, medium, high)
            5. Cultural emotional expression patterns
            6. Youth-specific emotional language indicators
            7. Secondary emotions if present
            
            Consider that emotional expression varies across cultures. In Rwandan culture:
            - Emotions may be expressed more indirectly
            - Family and community context affects emotional expression
            - Youth may use different emotional vocabulary
            - Cultural stigma may influence how emotions are communicated
            - Ubuntu philosophy emphasizes community and interconnectedness
            
            Available emotion response templates for {language}:
            {cultural_context.get('emotion_responses', '')}
            
            Respond in JSON format:
            {{
                "emotions": {{"anxiety": 0.7, "worry": 0.3, "neutral": 0.1}},
                "keywords": ["worried", "anxious", "cultural_expression"],
                "reasoning": "Text shows anxiety and worry with cultural context consideration",
                "selected_emotion": "anxiety",
                "confidence": 0.7,
                "intensity": "medium",
                "cultural_emotional_indicators": ["indirect_expression", "family_context"],
                "youth_emotional_patterns": ["peer_pressure", "academic_stress"],
                "secondary_emotions": ["worry", "uncertainty"],
                "cultural_appropriateness": "high"
            }}
            """
            
            user_prompt = f"""
            Analyze the emotions in this text with cultural awareness:
            
            Text: "{text}"
            Language: {language}
            Gender Context: {user_gender if user_gender else "not specified"}
            Additional Context: {context if context else "none"}
            
            Provide detailed emotion analysis with cultural sensitivity.
            """
            
            logger.info(f"📝 Emotion classification prompt prepared (system: {len(system_prompt)} chars)")
            
            # Make LLM call
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm_provider.generate_response(messages)
            logger.info(f"🤖 LLM response received: {len(response)} chars")
            
            # Parse response
            result = self._parse_emotion_response(response, text, language)
            logger.info(f"✅ Emotion classification completed: {result['selected_emotion']} (confidence: {result['confidence']:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Emotion classification failed: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "reasoning": f"Error during emotion classification: {str(e)}",
                "keywords": ["error", "fallback"],
                "cultural_context": {"language": "en", "error": str(e)}
            }
    
    def _parse_emotion_response(self, response: str, text: str, language: str) -> Dict[str, Any]:
        """Parse LLM response into structured emotion classification result."""
        try:
            # Try to parse JSON response
            data = json.loads(response)
            
            # Extract cultural elements
            cultural_indicators = data.get("cultural_emotional_indicators", [])
            youth_patterns = data.get("youth_emotional_patterns", [])
            keywords = data.get("keywords", [])
            secondary_emotions = data.get("secondary_emotions", [])
            
            # Combine all keywords
            all_keywords = keywords + cultural_indicators + youth_patterns + secondary_emotions
            
            return {
                "emotion": data.get("selected_emotion", "neutral"),
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": data.get("reasoning", "Emotion analysis completed with cultural context"),
                "keywords": all_keywords,
                "intensity": data.get("intensity", "medium"),
                "emotions": data.get("emotions", {"neutral": 1.0}),
                "cultural_context": {
                    "language": language,
                    "cultural_indicators": cultural_indicators,
                    "youth_patterns": youth_patterns,
                    "appropriateness": data.get("cultural_appropriateness", "medium")
                }
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback parsing
            return {
                "emotion": "neutral",
                "confidence": 0.3,
                "reasoning": "Fallback emotion classification due to parsing error",
                "keywords": ["cultural_context", "fallback"],
                "intensity": "low",
                "emotions": {"neutral": 1.0},
                "cultural_context": {
                    "language": language,
                    "error": "parsing_failed"
                }
            }

# Global LLM emotion classifier instance
_llm_emotion_classifier = None

def initialize_emotion_classifier(llm_provider: Optional[LLMProvider] = None) -> LLMEmotionClassifier:
    """Initialize the LLM emotion classifier."""
    global _llm_emotion_classifier
    if _llm_emotion_classifier is None:
        _llm_emotion_classifier = LLMEmotionClassifier(llm_provider)
        logger.info("🧠 LLM Emotion Classifier initialized globally")
    return _llm_emotion_classifier

# Backward compatibility function for simple emotion classification
async def classify_emotion(user_input: str, user_gender: Optional[str] = None) -> str:
    """
    Classify emotion using LLM analysis (backward compatibility function).
    
    Args:
        user_input: Text to analyze
        user_gender: User's gender for cultural context
        
    Returns:
        Primary emotion as string
    """
    global _llm_emotion_classifier
    
    if _llm_emotion_classifier is None:
        logger.warning("LLM emotion classifier not initialized, returning neutral")
        return "neutral"
    
    try:
        result = await _llm_emotion_classifier.classify_emotion(user_input, user_gender)
        return result.get("emotion", "neutral")
    except Exception as e:
        logger.error(f"Emotion classification failed: {e}")
        return "neutral"

# Synchronous wrapper for backward compatibility (fallback to neutral)
def classify_emotion_sync(user_input: str) -> str:
    """
    Synchronous emotion classification (fallback function).
    Returns neutral emotion for backward compatibility.
    """
    logger.warning("Using synchronous emotion classification fallback - returning neutral")
    return "neutral"