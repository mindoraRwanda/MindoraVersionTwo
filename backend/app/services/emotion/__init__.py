"""
Emotion Intelligence Service
============================
ML-powered emotion detection and crisis assessment for mental health support.
"""

from .schemas import (
    EmotionType,
    EmotionIntensity,
    EmotionResult,
    CrisisLevel,
    CrisisAssessment,
    MultiModalEmotion,
    TextEmotionRequest,
    CrisisAssessmentRequest
)

__all__ = [
    "EmotionType",
    "EmotionIntensity",
    "EmotionResult",
    "CrisisLevel",
    "CrisisAssessment",
    "MultiModalEmotion",
    "TextEmotionRequest",
    "CrisisAssessmentRequest"
]

__version__ = "1.0.0"
