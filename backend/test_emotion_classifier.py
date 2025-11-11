"""
Quick test script to verify emotion classifier is working correctly
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.services.emotion.text_emotion_classifier import TextEmotionClassifier

async def test_classifier():
    print("🧪 Testing Emotion Classifier\n")
    print("="*70)
    
    classifier = TextEmotionClassifier()
    
    test_cases = [
        ("Hello", "neutral or joy"),
        ("I am doing good", "joy"),
        ("I am exhausted", "sadness"),
        ("I am very happy!", "joy"),
        ("I feel angry", "anger"),
        ("Everything is overwhelming", "fear or sadness"),
        ("I want to die", "sadness or fear"),
    ]
    
    issues = []
    
    for text, expected in test_cases:
        result = await classifier.classify(text)
        
        # Check if detected emotion matches expected
        detected = result.primary_emotion.value.lower()
        expected_lower = expected.lower()
        
        is_correct = any(exp.strip() in detected for exp in expected_lower.split(' or '))
        status = "✅" if is_correct else "❌"
        
        if not is_correct:
            issues.append((text, expected, detected))
        
        print(f"{status} Text: '{text}'")
        print(f"   Expected: {expected}")
        print(f"   Got: {result.primary_emotion.value} (confidence: {result.confidence:.2f})")
        if result.secondary_emotions:
            secondary_str = ", ".join([f"{e.value}: {p:.2f}" for e, p in result.secondary_emotions.items()])
            print(f"   Secondary: {secondary_str}")
        print()
    
    print("="*70)
    if issues:
        print(f"\n❌ FOUND {len(issues)} ISSUES:\n")
        for text, expected, detected in issues:
            print(f"  • '{text}' -> Expected: {expected}, Got: {detected}")
        print("\n🐛 The emotion classifier needs debugging!")
    else:
        print("\n✅ All tests passed! Emotion classifier is working correctly!")

if __name__ == "__main__":
    asyncio.run(test_classifier())
