"""Test emotion schemas for data validation"""
import pytest
from datetime import datetime
from backend.app.services.emotion.schemas import (
    EmotionResult, EmotionType, EmotionIntensity,
    CrisisAssessment, CrisisLevel, TextEmotionRequest
)


def test_emotion_result_creation():
    """Valid emotion result should be created successfully"""
    result = EmotionResult(
        primary_emotion=EmotionType.SADNESS,
        intensity=EmotionIntensity.HIGH,
        confidence=0.85,
        secondary_emotions={EmotionType.FEAR: 0.25}
    )
    
    assert result.confidence == 0.85
    assert result.primary_emotion == EmotionType.SADNESS
    assert result.intensity == EmotionIntensity.HIGH
    assert EmotionType.FEAR in result.secondary_emotions
    assert isinstance(result.timestamp, datetime)


def test_invalid_confidence_rejected():
    """Confidence > 1.0 should raise validation error"""
    with pytest.raises(ValueError):
        EmotionResult(
            primary_emotion=EmotionType.JOY,
            intensity=EmotionIntensity.LOW,
            confidence=1.5  # Invalid!
        )


def test_invalid_secondary_emotion_score():
    """Secondary emotion scores must be 0-1"""
    with pytest.raises(ValueError):
        EmotionResult(
            primary_emotion=EmotionType.ANGER,
            intensity=EmotionIntensity.MEDIUM,
            confidence=0.7,
            secondary_emotions={EmotionType.FEAR: 1.5}  # Invalid!
        )


def test_crisis_assessment_structure():
    """Crisis assessment should contain all required fields"""
    emotion = EmotionResult(
        primary_emotion=EmotionType.DESPAIR,
        intensity=EmotionIntensity.CRITICAL,
        confidence=0.92
    )
    
    crisis = CrisisAssessment(
        level=CrisisLevel.CRITICAL,
        risk_score=0.95,
        triggers=["High despair (0.92)", "Critical intensity"],
        recommended_action="Escalate to counselor immediately. Call 114.",
        escalation_required=True,
        emotion_basis=emotion
    )
    
    assert crisis.escalation_required is True
    assert crisis.risk_score == 0.95
    assert crisis.level == CrisisLevel.CRITICAL
    assert len(crisis.triggers) == 2
    assert crisis.emotion_basis.primary_emotion == EmotionType.DESPAIR


def test_text_emotion_request_validation():
    """API request should validate text length"""
    # Valid request
    request = TextEmotionRequest(
        text="I feel sad",
        user_id="user123"
    )
    assert request.text == "I feel sad"
    
    # Empty text should fail
    with pytest.raises(ValueError):
        TextEmotionRequest(text="")
    
    # Text too long should fail
    with pytest.raises(ValueError):
        TextEmotionRequest(text="x" * 2001)


def test_emotion_enum_values():
    """Verify all emotion types are accessible"""
    assert EmotionType.SADNESS.value == "sadness"
    assert EmotionType.DESPAIR.value == "despair"
    assert EmotionType.JOY.value == "joy"
    
    # Test all intensity levels
    assert EmotionIntensity.CRITICAL.value == "critical"
    assert EmotionIntensity.LOW.value == "low"
