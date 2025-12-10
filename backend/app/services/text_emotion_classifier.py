import logging
from typing import Dict, List, Optional, Tuple
import torch
from transformers import pipeline
import re

logger = logging.getLogger(__name__)

class TextEmotionClassifier:
    """
    Hybrid emotion classifier combining ML (DistilRoBERTa) with 
    cultural rule-based boosting for Rwandan context.
    """
    
    # Cultural distress markers in Kinyarwanda/English mix
    RWANDAN_DISTRESS_MARKERS = {
        "mfite ikibazo": ("fear", 1.5),      # "I have a problem" -> Anxiety/Fear
        "naniwe": ("sadness", 1.5),          # "I am tired" (often means depressed) -> Sadness
        "gupfa": ("sadness", 2.0),           # "Die/Death" -> Critical Sadness
        "ubwoba": ("fear", 1.5),             # "Fear" -> Fear
        "agahinda": ("sadness", 1.5),        # "Sorrow/Grief" -> Sadness
        "umujinya": ("anger", 1.5),          # "Anger" -> Anger
        "wihebye": ("sadness", 1.8),         # "Despair" -> Sadness
        "stress": ("fear", 1.2),             # Common loan word
        "depressed": ("sadness", 1.5),       # Common loan word
        "anxiety": ("fear", 1.5),            # Common loan word
    }

    # Intensity modifiers
    INTENSIFIERS = {
        "cyane": 1.3,        # "Very/A lot"
        "rwose": 1.4,        # "Truly/Really"
        "kabisa": 1.3,       # "Completely"
        "very": 1.3,
        "extremely": 1.5,
        "so": 1.2,
        "really": 1.3,
        "totally": 1.3
    }

    def __init__(self):
        self.classifier = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the HuggingFace emotion model."""
        try:
            # Using a model fine-tuned for emotion detection
            model_name = "j-hartmann/emotion-english-distilroberta-base"
            device = 0 if torch.cuda.is_available() else -1
            
            self.classifier = pipeline(
                "text-classification", 
                model=model_name, 
                top_k=None,  # Return all scores
                device=device
            )
            logger.info(f"Emotion classifier initialized on device {device}")
        except Exception as e:
            logger.error(f"Failed to initialize emotion model: {e}")
            # Fallback or handle error appropriately
            self.classifier = None

    def detect_emotion(self, text: str) -> Dict[str, any]:
        """
        Detect emotion using hybrid approach: ML + Cultural Rules.
        """
        if not text:
            return self._get_default_response()

        # 1. Get Base ML Scores
        ml_scores = self._get_ml_scores(text)
        
        # 2. Apply Cultural Boosting
        final_scores, detected_markers = self._apply_cultural_boosting(text, ml_scores)
        
        # 3. Determine Primary Emotion & Intensity
        primary_emotion = max(final_scores.items(), key=lambda x: x[1])
        intensity = self._calculate_intensity(text, primary_emotion[1])

        return {
            "selected_emotion": primary_emotion[0],
            "confidence": float(primary_emotion[1]),
            "all_scores": final_scores,
            "intensity": intensity,
            "cultural_markers": detected_markers,
            "is_crisis": primary_emotion[0] in ["sadness", "fear"] and primary_emotion[1] > 0.8
        }

    def _get_ml_scores(self, text: str) -> Dict[str, float]:
        """Get raw scores from the transformer model."""
        if not self.classifier:
            return {"neutral": 1.0}

        try:
            # Truncate text if too long for model
            results = self.classifier(text[:512])
            # Convert list of dicts to dict of scores
            scores = {item['label']: item['score'] for item in results[0]}
            return scores
        except Exception as e:
            logger.error(f"ML inference failed: {e}")
            return {"neutral": 1.0}

    def _apply_cultural_boosting(self, text: str, scores: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
        """Boost scores based on Rwandan cultural markers."""
        text_lower = text.lower()
        boosted_scores = scores.copy()
        detected_markers = []

        for marker, (emotion, boost_factor) in self.RWANDAN_DISTRESS_MARKERS.items():
            if marker in text_lower:
                detected_markers.append(marker)
                # Map model labels to our internal labels if needed
                # Model labels: anger, disgust, fear, joy, neutral, sadness, surprise
                target_emotion = emotion
                
                if target_emotion in boosted_scores:
                    boosted_scores[target_emotion] *= boost_factor
                    
                    # Dampen opposite emotions
                    if target_emotion == "sadness":
                        boosted_scores["joy"] *= 0.5
                    elif target_emotion == "joy":
                        boosted_scores["sadness"] *= 0.5

        # Re-normalize scores to sum to ~1 (optional, but good for consistency)
        total = sum(boosted_scores.values())
        if total > 0:
            boosted_scores = {k: v/total for k, v in boosted_scores.items()}

        return boosted_scores, detected_markers

    def _calculate_intensity(self, text: str, score: float) -> str:
        """Determine intensity level based on score and intensifiers."""
        text_lower = text.lower()
        intensity_mult = 1.0
        
        for word, mult in self.INTENSIFIERS.items():
            if word in text_lower:
                intensity_mult *= mult

        final_score = score * intensity_mult

        if final_score > 0.85:
            return "CRITICAL"
        elif final_score > 0.7:
            return "HIGH"
        elif final_score > 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_default_response(self):
        return {
            "selected_emotion": "neutral",
            "confidence": 1.0,
            "all_scores": {"neutral": 1.0},
            "intensity": "LOW",
            "cultural_markers": [],
            "is_crisis": False
        }
