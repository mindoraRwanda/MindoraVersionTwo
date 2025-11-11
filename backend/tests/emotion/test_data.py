"""
Test data for emotion classification
=====================================
Real-world test cases for emotion detection accuracy validation.

Includes:
- Standard English emotion examples
- Kinyarwanda-English hybrid cases
- Crisis scenarios
- Edge cases (sarcasm, ambiguity)
"""

from backend.app.services.emotion.schemas import EmotionType, EmotionIntensity, CrisisLevel


# Emotion classification test cases
EMOTION_TEST_CASES = [
    # ============ JOY ============
    {
        "text": "I'm so happy today! Everything is going well.",
        "expected_emotion": EmotionType.JOY,
        "expected_intensity": EmotionIntensity.MEDIUM,
        "min_confidence": 0.7,
        "description": "Clear positive emotion"
    },
    {
        "text": "I feel amazing! Life is beautiful!",
        "expected_emotion": EmotionType.JOY,
        "expected_intensity": EmotionIntensity.HIGH,
        "min_confidence": 0.75
    },
    
    # ============ SADNESS ============
    {
        "text": "I feel so sad and alone. Nobody understands me.",
        "expected_emotion": EmotionType.SADNESS,
        "expected_intensity": EmotionIntensity.HIGH,
        "min_confidence": 0.7,
        "description": "High sadness with isolation"
    },
    {
        "text": "I've been feeling down lately.",
        "expected_emotion": EmotionType.SADNESS,
        "expected_intensity": EmotionIntensity.MEDIUM,
        "min_confidence": 0.6
    },
    {
        "text": "Everything feels heavy. I don't have energy for anything.",
        "expected_emotion": EmotionType.SADNESS,
        "expected_intensity": EmotionIntensity.HIGH,
        "min_confidence": 0.65
    },
    
    # ============ DESPAIR (Critical) ============
    {
        "text": "I can't take this anymore. I want to end it all.",
        "expected_emotion": EmotionType.DESPAIR,
        "expected_intensity": EmotionIntensity.CRITICAL,
        "min_confidence": 0.7,
        "crisis_level": CrisisLevel.CRITICAL,
        "description": "Suicidal ideation - immediate escalation"
    },
    {
        "text": "There's no point in living. I'm worthless.",
        "expected_emotion": EmotionType.DESPAIR,
        "expected_intensity": EmotionIntensity.CRITICAL,
        "min_confidence": 0.7,
        "crisis_level": CrisisLevel.CRITICAL
    },
    {
        "text": "I feel hopeless. Nothing will ever get better.",
        "expected_emotion": EmotionType.DESPAIR,
        "expected_intensity": EmotionIntensity.HIGH,
        "min_confidence": 0.65,
        "crisis_level": CrisisLevel.HIGH
    },
    
    # ============ FEAR/ANXIETY ============
    {
        "text": "I'm so worried about the exam tomorrow. I can't sleep.",
        "expected_emotion": EmotionType.ANXIETY,
        "expected_intensity": EmotionIntensity.MEDIUM,
        "min_confidence": 0.6
    },
    {
        "text": "I'm scared something bad will happen to my family.",
        "expected_emotion": EmotionType.FEAR,
        "expected_intensity": EmotionIntensity.HIGH,
        "min_confidence": 0.65
    },
    {
        "text": "My heart won't stop racing. I feel like I'm dying.",
        "expected_emotion": EmotionType.ANXIETY,
        "expected_intensity": EmotionIntensity.HIGH,
        "min_confidence": 0.7,
        "description": "Panic attack symptoms"
    },
    
    # ============ ANGER ============
    {
        "text": "I'm so angry at them! This is completely unfair!",
        "expected_emotion": EmotionType.ANGER,
        "expected_intensity": EmotionIntensity.HIGH,
        "min_confidence": 0.7
    },
    {
        "text": "I'm frustrated with how things are going.",
        "expected_emotion": EmotionType.ANGER,
        "expected_intensity": EmotionIntensity.MEDIUM,
        "min_confidence": 0.6
    },
    
    # ============ KINYARWANDA-ENGLISH HYBRID ============
    {
        "text": "Ndi sad cyane, I don't know what to do",
        "expected_emotion": EmotionType.SADNESS,
        "expected_intensity": EmotionIntensity.HIGH,
        "cultural_marker": "ndi",
        "min_confidence": 0.6,
        "description": "Kinyarwanda marker: 'ndi' (I am)"
    },
    {
        "text": "Mfite ikibazo and I feel hopeless about it",
        "expected_emotion": EmotionType.DESPAIR,
        "expected_intensity": EmotionIntensity.HIGH,
        "cultural_marker": "mfite ikibazo",
        "min_confidence": 0.65,
        "description": "Kinyarwanda: 'I have a problem'"
    },
    {
        "text": "Ndababaye cyane, I can't go on living like this",
        "expected_emotion": EmotionType.DESPAIR,
        "expected_intensity": EmotionIntensity.CRITICAL,
        "cultural_marker": "ndababaye",
        "min_confidence": 0.7,
        "crisis_level": CrisisLevel.HIGH,
        "description": "Kinyarwanda: 'I'm suffering'"
    },
    {
        "text": "Ntacyo nshobora gukora, everything is falling apart",
        "expected_emotion": EmotionType.DESPAIR,
        "expected_intensity": EmotionIntensity.HIGH,
        "cultural_marker": "ntacyo nshobora gukora",
        "min_confidence": 0.65,
        "description": "Kinyarwanda: 'I can't do anything'"
    },
    
    # ============ NEUTRAL ============
    {
        "text": "I went to school today and came back home.",
        "expected_emotion": EmotionType.NEUTRAL,
        "expected_intensity": EmotionIntensity.LOW,
        "min_confidence": 0.5,
        "description": "Factual statement, no emotion"
    },
    
    # ============ AMBIGUOUS/EDGE CASES ============
    {
        "text": "I'm fine.",
        "expected_emotion": EmotionType.NEUTRAL,
        "expected_intensity": EmotionIntensity.LOW,
        "min_confidence": 0.4,
        "description": "Could be sarcasm or genuine - needs context"
    },
    {
        "text": "Whatever.",
        "expected_emotion": EmotionType.NEUTRAL,
        "expected_intensity": EmotionIntensity.LOW,
        "min_confidence": 0.3,
        "description": "Ambiguous single word"
    },
]


# Crisis detection test cases
CRISIS_TEST_CASES = [
    # ============ CRITICAL LEVEL ============
    {
        "text": "I'm thinking about suicide",
        "expected_crisis_level": CrisisLevel.CRITICAL,
        "expected_escalation": True,
        "min_risk_score": 0.85,
        "triggers": ["Critical keywords", "High despair"]
    },
    {
        "text": "I want to kill myself",
        "expected_crisis_level": CrisisLevel.CRITICAL,
        "expected_escalation": True,
        "min_risk_score": 0.9,
        "triggers": ["Critical keywords"]
    },
    {
        "text": "There's no point in living anymore. I want to end it.",
        "expected_crisis_level": CrisisLevel.CRITICAL,
        "expected_escalation": True,
        "min_risk_score": 0.85,
        "triggers": ["Critical keywords", "Hopelessness"]
    },
    
    # ============ HIGH LEVEL ============
    {
        "text": "I feel completely hopeless. Nothing matters anymore.",
        "expected_crisis_level": CrisisLevel.HIGH,
        "expected_escalation": True,
        "min_risk_score": 0.65,
        "triggers": ["High despair", "Hopelessness indicators"]
    },
    {
        "text": "Ndababaye cyane, I can't continue like this",
        "expected_crisis_level": CrisisLevel.HIGH,
        "expected_escalation": True,
        "min_risk_score": 0.65,
        "triggers": ["Kinyarwanda distress marker", "High intensity"]
    },
    
    # ============ MEDIUM LEVEL ============
    {
        "text": "I've been feeling really sad for weeks. I don't enjoy anything anymore.",
        "expected_crisis_level": CrisisLevel.MEDIUM,
        "expected_escalation": False,
        "min_risk_score": 0.45,
        "triggers": ["Persistent sadness", "Anhedonia"]
    },
    {
        "text": "Sometimes I think things would be easier if I wasn't here.",
        "expected_crisis_level": CrisisLevel.MEDIUM,
        "expected_escalation": False,
        "min_risk_score": 0.5,
        "triggers": ["Passive suicidal ideation"]
    },
    
    # ============ LOW LEVEL ============
    {
        "text": "I'm feeling down today, but I'll be okay.",
        "expected_crisis_level": CrisisLevel.LOW,
        "expected_escalation": False,
        "min_risk_score": 0.25,
        "triggers": []
    },
    
    # ============ NONE ============
    {
        "text": "I had a great day! Feeling happy.",
        "expected_crisis_level": CrisisLevel.NONE,
        "expected_escalation": False,
        "min_risk_score": 0.1,
        "triggers": []
    },
]


# Test data for conversation trajectory analysis
CONVERSATION_TRAJECTORY_TESTS = [
    {
        "description": "Deteriorating emotional state",
        "messages": [
            {"text": "I'm okay, just tired", "expected_emotion": EmotionType.NEUTRAL},
            {"text": "I've been feeling down lately", "expected_emotion": EmotionType.SADNESS},
            {"text": "I don't think things will get better", "expected_emotion": EmotionType.DESPAIR}
        ],
        "expected_trend": "worsening",
        "expected_risk_boost": 0.3,
        "description_detail": "Should detect increasing negativity over 3 messages"
    },
    {
        "description": "Improving emotional state",
        "messages": [
            {"text": "I feel hopeless", "expected_emotion": EmotionType.DESPAIR},
            {"text": "Talking to you helps a bit", "expected_emotion": EmotionType.SADNESS},
            {"text": "Maybe I can try one more thing", "expected_emotion": EmotionType.HOPE}
        ],
        "expected_trend": "improving",
        "expected_risk_boost": -0.2,
        "description_detail": "Should detect emotional recovery"
    },
]
