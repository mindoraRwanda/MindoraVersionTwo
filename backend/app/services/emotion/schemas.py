"""
Emotion Analysis Data Models
=============================
Pydantic schemas for emotion classification results, crisis assessment,
and multi-modal emotion fusion.

Author: Mindora Team
Date: November 2025
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List
from enum import Enum
from datetime import datetime


class EmotionType(str, Enum):
    """
    Core emotion types based on Ekman's basic emotions + mental health context.
    Extended for therapy/crisis scenarios.
    """
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    ANXIETY = "anxiety"      # Mental health specific
    DESPAIR = "despair"      # Critical for crisis detection
    HOPE = "hope"            # Positive mental health indicator
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"      # Fallback for low confidence


class EmotionIntensity(str, Enum):
    """
    Emotion intensity levels for crisis stratification.
    Maps to intervention urgency.
    """
    LOW = "low"              # Normal conversation
    MEDIUM = "medium"        # Elevated concern
    HIGH = "high"            # Needs attention
    CRITICAL = "critical"    # Immediate intervention required


class CrisisLevel(str, Enum):
    """
    Crisis severity stratification for escalation decisions.
    Based on Liu et al. (2023) multi-step assessment framework.
    """
    NONE = "none"            # No crisis indicators
    LOW = "low"              # Minor distress, monitor
    MEDIUM = "medium"        # Moderate risk, increase support
    HIGH = "high"            # Serious risk, alert therapist
    CRITICAL = "critical"    # Immediate danger, escalate to counselor


class EmotionResult(BaseModel):
    """
    Comprehensive emotion classification result from ML model.
    
    Attributes:
        primary_emotion: Dominant emotion detected
        intensity: Severity level (low → critical)
        confidence: Model certainty (0.0-1.0)
        secondary_emotions: Additional emotions with scores
        cultural_context: Detected cultural markers (e.g., Kinyarwanda phrases)
        timestamp: When emotion was detected (for trajectory tracking)
        model_version: ML model identifier for auditing
    """
    primary_emotion: EmotionType
    intensity: EmotionIntensity
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    secondary_emotions: Dict[EmotionType, float] = Field(
        default_factory=dict,
        description="Mixed emotions with scores (e.g., {FEAR: 0.25, SADNESS: 0.15})"
    )
    cultural_context: Optional[str] = Field(
        default=None,
        description="Detected cultural markers (e.g., 'Kinyarwanda: ndi sad')"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = Field(
        default="emotion-distilroberta-v1",
        description="Model identifier for version tracking"
    )
    
    @field_validator('secondary_emotions')
    @classmethod
    def validate_secondary_scores(cls, v: Dict[EmotionType, float]) -> Dict[EmotionType, float]:
        """Ensure all secondary emotion scores are valid probabilities"""
        for emotion, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Invalid emotion score for {emotion}: {score}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "primary_emotion": "sadness",
                "intensity": "high",
                "confidence": 0.82,
                "secondary_emotions": {
                    "fear": 0.25,
                    "despair": 0.18
                },
                "cultural_context": "Kinyarwanda marker: ndababaye",
                "timestamp": "2025-11-03T14:23:11.123456",
                "model_version": "emotion-distilroberta-v1"
            }
        }


class CrisisAssessment(BaseModel):
    """
    Multi-step crisis risk assessment result.
    Combines emotion signals, keywords, and conversation history.
    
    Based on cascaded classification approach (Cruz-Gonzalez et al., 2023):
    Emotion → Risk Scoring → Severity Stratification → Escalation Decision
    
    Attributes:
        level: Crisis severity (none → critical)
        risk_score: Fused risk probability (0.0-1.0)
        triggers: List of detected risk factors
        recommended_action: Human-readable intervention guidance
        escalation_required: Whether to immediately escalate to counselor
        emotion_basis: The emotion result that triggered assessment
        temporal_trend: Emotional trajectory score (if history available)
    """
    level: CrisisLevel
    risk_score: float = Field(
        ge=0.0, le=1.0,
        description="Fused crisis probability from multi-signal analysis"
    )
    triggers: List[str] = Field(
        default_factory=list,
        description="Detected risk factors (e.g., ['High despair', 'Critical keywords'])"
    )
    recommended_action: str = Field(
        description="Human-readable intervention guidance"
    )
    escalation_required: bool = Field(
        description="True if immediate counselor escalation needed"
    )
    emotion_basis: EmotionResult = Field(
        description="The emotion detection that triggered this assessment"
    )
    temporal_trend: Optional[float] = Field(
        default=None,
        ge=-1.0, le=1.0,
        description="Emotion trajectory: negative=improving, positive=worsening"
    )
    assessment_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "level": "high",
                "risk_score": 0.78,
                "triggers": [
                    "High-risk emotion: despair (confidence=0.85)",
                    "Critical intensity level",
                    "Deteriorating emotional state over conversation"
                ],
                "recommended_action": "Alert on-call therapist. Provide crisis hotline (114).",
                "escalation_required": True,
                "emotion_basis": {
                    "primary_emotion": "despair",
                    "intensity": "critical",
                    "confidence": 0.85
                },
                "temporal_trend": 0.65
            }
        }


class MultiModalEmotion(BaseModel):
    """
    Fused emotion from text + voice analysis (Phase 2).
    Combines modalities for higher accuracy.
    
    Research shows 15-25% accuracy improvement with multi-modal fusion
    (Wang et al., 2023).
    
    Attributes:
        text_emotion: Emotion from text analysis
        voice_emotion: Emotion from acoustic features (optional, Phase 2)
        fused_emotion: Combined result (weighted average based on confidence)
        modality_agreement: How aligned are text/voice emotions (0-1)
        fusion_method: Algorithm used (e.g., 'confidence_weighted')
    """
    text_emotion: EmotionResult
    voice_emotion: Optional[EmotionResult] = Field(
        default=None,
        description="Voice-based emotion (Phase 2 - acoustic features)"
    )
    fused_emotion: EmotionResult = Field(
        description="Final fused emotion result"
    )
    modality_agreement: float = Field(
        ge=0.0, le=1.0,
        description="Cross-modal consistency score (1.0 = perfect agreement)"
    )
    fusion_method: str = Field(
        default="confidence_weighted",
        description="Fusion algorithm: 'confidence_weighted' or 'majority_vote'"
    )
    
    @field_validator('modality_agreement')
    @classmethod
    def validate_agreement(cls, v: float, info) -> float:
        """Agreement only meaningful when both modalities present"""
        if info.data.get('voice_emotion') is None and v != 1.0:
            raise ValueError("Agreement must be 1.0 when only text modality used")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "text_emotion": {
                    "primary_emotion": "sadness",
                    "confidence": 0.72
                },
                "voice_emotion": {
                    "primary_emotion": "despair",
                    "confidence": 0.88
                },
                "fused_emotion": {
                    "primary_emotion": "despair",
                    "confidence": 0.82,
                    "secondary_emotions": {"sadness": 0.35}
                },
                "modality_agreement": 0.75,
                "fusion_method": "confidence_weighted"
            }
        }


class TextEmotionRequest(BaseModel):
    """API request schema for text emotion analysis"""
    text: str = Field(
        min_length=1,
        max_length=2000,
        description="User message (supports Kinyarwanda-English hybrid)"
    )
    context: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Optional conversation context for better accuracy"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier for conversation history tracking"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Ndi sad cyane, I can't handle this anymore",
                "context": "Previous message: 'I've been feeling down for weeks'",
                "user_id": "user_12345"
            }
        }


class CrisisAssessmentRequest(BaseModel):
    """API request schema for crisis assessment"""
    text: str = Field(min_length=1, max_length=2000)
    emotion: EmotionResult = Field(
        description="Pre-computed emotion (from text_emotion endpoint)"
    )
    conversation_history: Optional[List[EmotionResult]] = Field(
        default=None,
        max_items=10,
        description="Recent emotion results for trajectory analysis"
    )
    user_id: Optional[str] = Field(default=None)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I want to end it all",
                "emotion": {
                    "primary_emotion": "despair",
                    "intensity": "critical",
                    "confidence": 0.92
                },
                "conversation_history": [
                    {"primary_emotion": "sadness", "timestamp": "2025-11-03T14:00:00"},
                    {"primary_emotion": "despair", "timestamp": "2025-11-03T14:15:00"}
                ]
            }
        }
