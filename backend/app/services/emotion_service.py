"""
Unified Emotion Classification Service

This module provides a unified interface for emotion classification that can use
multiple models (DistilBERT, similarity-based) with fallback mechanisms and
hybrid approaches for improved accuracy and reliability.
"""

import logging
from typing import Dict, List, Optional, Any
import asyncio
from enum import Enum

from .emotion_classifier_v2 import EmotionClassifierV2, get_emotion_classifier_v2
from .emotion_classifier import classify_emotion as classify_emotion_legacy, initialize_emotion_classifier
from .emotion_config import get_emotion_config, EmotionClassifierType

logger = logging.getLogger(__name__)

class EmotionService:
    """
    Unified emotion classification service with multiple model support.
    
    This service can use different emotion classification approaches:
    - DistilBERT: Advanced transformer-based classification
    - Similarity: Embedding-based cosine similarity approach  
    - Hybrid: Combination of both methods with confidence voting
    """
    
    def __init__(self):
        """Initialize the emotion service."""
        self.config = get_emotion_config()
        self.distilbert_classifier: Optional[EmotionClassifierV2] = None
        self.similarity_classifier_initialized = False
        self.is_initialized = False
        
        logger.info(f"Initializing EmotionService with type: {self.config.classifier_type.value}")
    
    async def initialize(self) -> bool:
        """
        Initialize the emotion classifiers based on configuration.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            classifier_type = self.config.classifier_type
            
            if classifier_type in [EmotionClassifierType.DISTILBERT, EmotionClassifierType.HYBRID]:
                logger.info("Initializing DistilBERT classifier...")
                self.distilbert_classifier = await get_emotion_classifier_v2()
                
            if classifier_type in [EmotionClassifierType.SIMILARITY, EmotionClassifierType.HYBRID]:
                logger.info("Initializing similarity classifier...")
                initialize_emotion_classifier()
                self.similarity_classifier_initialized = True
            
            self.is_initialized = True
            logger.info("EmotionService initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize EmotionService: {str(e)}")
            return False
    
    async def classify_emotion(self, text: str) -> Dict[str, Any]:
        """
        Classify emotion using the configured approach.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dict containing emotion classification results
        """
        if not self.is_initialized:
            await self.initialize()
        
        classifier_type = self.config.classifier_type
        
        try:
            if classifier_type == EmotionClassifierType.DISTILBERT:
                return await self._classify_distilbert(text)
            
            elif classifier_type == EmotionClassifierType.SIMILARITY:
                return await self._classify_similarity(text)
            
            elif classifier_type == EmotionClassifierType.HYBRID:
                return await self._classify_hybrid(text)
            
            else:
                logger.warning(f"Unknown classifier type: {classifier_type}")
                return self._get_fallback_result(text)
                
        except Exception as e:
            logger.error(f"Error in emotion classification: {str(e)}")
            return self._get_fallback_result(text, error=str(e))
    
    async def _classify_distilbert(self, text: str) -> Dict[str, Any]:
        """Classify using DistilBERT model."""
        if not self.distilbert_classifier:
            logger.error("DistilBERT classifier not initialized")
            return self._get_fallback_result(text)
        
        result = await self.distilbert_classifier.classify_emotion(text)
        result["service"] = "distilbert"
        return result
    
    async def _classify_similarity(self, text: str) -> Dict[str, Any]:
        """Classify using similarity-based approach."""
        if not self.similarity_classifier_initialized:
            logger.error("Similarity classifier not initialized")
            return self._get_fallback_result(text)
        
        # Use the function-based legacy classifier
        predicted_emotion = classify_emotion_legacy(text)
        
        # Normalize result format to match DistilBERT output
        normalized_result = {
            "emotion": predicted_emotion,
            "confidence": 0.7,  # Default confidence for legacy classifier
            "all_scores": {predicted_emotion: 0.7},
            "method": "similarity",
            "service": "similarity"
        }
        
        return normalized_result
    
    async def _classify_hybrid(self, text: str) -> Dict[str, Any]:
        """
        Classify using hybrid approach combining both methods.
        
        Uses weighted voting between DistilBERT and similarity approaches
        with fallback mechanisms for improved reliability.
        """
        distilbert_result = None
        similarity_result = None
        
        # Get predictions from both models
        try:
            if self.distilbert_classifier:
                distilbert_result = await self._classify_distilbert(text)
        except Exception as e:
            logger.warning(f"DistilBERT classification failed: {str(e)}")
        
        try:
            if self.similarity_classifier_initialized:
                similarity_result = await self._classify_similarity(text)
        except Exception as e:
            logger.warning(f"Similarity classification failed: {str(e)}")
        
        # Handle cases where one or both methods failed
        if not distilbert_result and not similarity_result:
            return self._get_fallback_result(text, error="Both classifiers failed")
        
        if not distilbert_result:
            similarity_result["service"] = "hybrid_similarity_only"
            return similarity_result
        
        if not similarity_result:
            distilbert_result["service"] = "hybrid_distilbert_only"
            return distilbert_result
        
        # Combine results using weighted voting
        return self._combine_predictions(distilbert_result, similarity_result, text)
    
    def _combine_predictions(self, distilbert_result: Dict, similarity_result: Dict, text: str) -> Dict[str, Any]:
        """
        Combine predictions from both classifiers using weighted voting.
        
        Args:
            distilbert_result: Results from DistilBERT classifier
            similarity_result: Results from similarity classifier
            text: Original input text
            
        Returns:
            Combined prediction result
        """
        try:
            # Get weights from configuration
            weight_distilbert = self.config.hybrid_weight_distilbert
            weight_similarity = self.config.hybrid_weight_similarity
            
            # Get predictions
            distilbert_emotion = distilbert_result.get("emotion", self.config.fallback_emotion)
            distilbert_confidence = distilbert_result.get("confidence", 0.0)
            
            similarity_emotion = similarity_result.get("emotion", self.config.fallback_emotion)
            similarity_confidence = similarity_result.get("confidence", 0.0)
            
            # Check if both models agree
            models_agree = distilbert_emotion == similarity_emotion
            
            # Calculate weighted confidence
            if models_agree:
                # If models agree, boost confidence
                final_emotion = distilbert_emotion
                final_confidence = min(1.0, 
                    (distilbert_confidence * weight_distilbert + 
                     similarity_confidence * weight_similarity) * 1.2)
            else:
                # If models disagree, use higher confidence weighted result
                distilbert_weighted = distilbert_confidence * weight_distilbert
                similarity_weighted = similarity_confidence * weight_similarity
                
                if distilbert_weighted > similarity_weighted:
                    final_emotion = distilbert_emotion
                    final_confidence = distilbert_confidence * 0.8  # Reduce confidence due to disagreement
                else:
                    final_emotion = similarity_emotion
                    final_confidence = similarity_confidence * 0.8
            
            # Apply minimum agreement threshold
            if final_confidence < self.config.hybrid_min_agreement:
                final_emotion = self.config.fallback_emotion
                final_confidence = self.config.hybrid_min_agreement
            
            # Combine all scores
            all_scores = {}
            distilbert_scores = distilbert_result.get("all_scores", {})
            similarity_scores = similarity_result.get("all_scores", {})
            
            # Merge scores with weighting
            all_emotions = set(list(distilbert_scores.keys()) + list(similarity_scores.keys()))
            for emotion in all_emotions:
                distilbert_score = distilbert_scores.get(emotion, 0.0)
                similarity_score = similarity_scores.get(emotion, 0.0)
                all_scores[emotion] = (
                    distilbert_score * weight_distilbert + 
                    similarity_score * weight_similarity
                )
            
            return {
                "emotion": final_emotion,
                "confidence": float(final_confidence),
                "all_scores": all_scores,
                "method": "hybrid",
                "service": "hybrid",
                "models_agree": models_agree,
                "distilbert_prediction": {
                    "emotion": distilbert_emotion,
                    "confidence": distilbert_confidence
                },
                "similarity_prediction": {
                    "emotion": similarity_emotion, 
                    "confidence": similarity_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Error combining predictions: {str(e)}")
            return self._get_fallback_result(text, error=str(e))
    
    def _get_fallback_result(self, text: str, error: Optional[str] = None) -> Dict[str, Any]:
        """Get fallback result when classification fails."""
        result = {
            "emotion": self.config.fallback_emotion,
            "confidence": 0.0,
            "all_scores": {self.config.fallback_emotion: 1.0},
            "method": "fallback",
            "service": "fallback"
        }
        
        if error:
            result["error"] = error
            
        return result
    
    async def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify emotions for multiple texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of emotion classification results
        """
        if not texts:
            return []
        
        results = []
        for text in texts:
            result = await self.classify_emotion(text)
            results.append(result)
        
        return results
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the emotion service configuration."""
        return {
            "classifier_type": self.config.classifier_type.value,
            "is_initialized": self.is_initialized,
            "available_classifiers": {
                "distilbert": self.distilbert_classifier is not None,
                "similarity": self.similarity_classifier_initialized
            },
            "configuration": self.config.to_dict()
        }

# Global service instance
_emotion_service: Optional[EmotionService] = None

async def get_emotion_service() -> EmotionService:
    """
    Get or create the global emotion service instance.
    
    Returns:
        EmotionService instance
    """
    global _emotion_service
    
    if _emotion_service is None:
        _emotion_service = EmotionService()
        await _emotion_service.initialize()
    
    return _emotion_service

async def initialize_emotion_service() -> bool:
    """
    Initialize the global emotion service.
    
    Returns:
        bool: True if successful
    """
    try:
        service = await get_emotion_service()
        logger.info("EmotionService initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize EmotionService: {str(e)}")
        return False

# Convenience function for easy classification
async def classify_emotion(text: str) -> Dict[str, Any]:
    """
    Classify emotion using the configured service.
    
    Args:
        text: Input text to classify
        
    Returns:
        Dict with emotion classification results
    """
    service = await get_emotion_service()
    return await service.classify_emotion(text)