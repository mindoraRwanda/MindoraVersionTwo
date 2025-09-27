"""
Test script for the new emotion classification pipeline.

This script tests the DistilBERT-based emotion classifier and compares
it with the legacy similarity-based approach.
"""

import asyncio
import time
from typing import List, Dict, Any

# Test samples covering different emotions
TEST_SAMPLES = [
    # Joy/Happiness
    {
        "text": "I'm feeling so happy today! Everything seems to be going perfectly.",
        "expected": "happiness"
    },
    {
        "text": "I just got promoted at work and I'm over the moon!",
        "expected": "happiness"
    },
    
    # Sadness
    {
        "text": "I've been feeling really down lately and nothing seems to help.",
        "expected": "sadness"
    },
    {
        "text": "My dog passed away last week and I can't stop crying.",
        "expected": "sadness"
    },
    
    # Anger
    {
        "text": "I'm so frustrated with this situation, it's driving me crazy!",
        "expected": "anger"
    },
    {
        "text": "This is absolutely infuriating, I can't believe this happened.",
        "expected": "anger"
    },
    
    # Fear/Anxiety
    {
        "text": "I'm really worried about my job interview tomorrow.",
        "expected": "fear"
    },
    {
        "text": "I keep having panic attacks and I'm scared it will happen again.",
        "expected": "fear"
    },
    
    # Neutral
    {
        "text": "Hello, how are you doing today?",
        "expected": "neutral"
    },
    {
        "text": "I went to the store to buy some groceries.",
        "expected": "neutral"
    },
    
    # Disgust
    {
        "text": "That behavior is absolutely disgusting and unacceptable.",
        "expected": "disgust"
    },
    
    # Surprise
    {
        "text": "I can't believe they threw me a surprise party!",
        "expected": "surprise"
    }
]

async def test_emotion_classifier_v2():
    """Test the new DistilBERT-based emotion classifier."""
    print("🚀 Testing Enhanced Emotion Classifier (DistilBERT)")
    print("=" * 60)
    
    try:
        from backend.app.services.emotion_service import classify_emotion, get_emotion_service
        
        # Get service info
        service = await get_emotion_service()
        service_info = service.get_service_info()
        
        print(f"📊 Service Configuration:")
        print(f"   Classifier Type: {service_info['classifier_type']}")
        print(f"   Initialized: {service_info['is_initialized']}")
        print(f"   Available Classifiers: {service_info['available_classifiers']}")
        print()
        
        results = []
        total_time = 0
        
        for i, sample in enumerate(TEST_SAMPLES, 1):
            text = sample["text"]
            expected = sample["expected"]
            
            print(f"Test {i:2d}: {text[:50]}...")
            
            # Time the classification
            start_time = time.time()
            result = await classify_emotion(text)
            end_time = time.time()
            
            classification_time = end_time - start_time
            total_time += classification_time
            
            predicted_emotion = result.get("emotion", "unknown")
            confidence = result.get("confidence", 0.0)
            method = result.get("method", "unknown")
            
            # Check if prediction matches expected
            is_correct = predicted_emotion == expected
            
            print(f"         Expected: {expected}")
            print(f"         Predicted: {predicted_emotion} (confidence: {confidence:.3f})")
            print(f"         Method: {method}")
            print(f"         Time: {classification_time:.3f}s")
            print(f"         ✅ Correct" if is_correct else f"         ❌ Incorrect")
            print()
            
            results.append({
                "text": text,
                "expected": expected,
                "predicted": predicted_emotion,
                "confidence": confidence,
                "method": method,
                "time": classification_time,
                "correct": is_correct,
                "full_result": result
            })
        
        # Calculate metrics
        correct_predictions = sum(1 for r in results if r["correct"])
        total_predictions = len(results)
        accuracy = correct_predictions / total_predictions
        avg_time = total_time / total_predictions
        avg_confidence = sum(r["confidence"] for r in results) / total_predictions
        
        print("📈 PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.2%}")
        print(f"Average Time: {avg_time:.3f}s per classification")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Total Time: {total_time:.3f}s")
        print()
        
        # Show incorrect predictions
        incorrect = [r for r in results if not r["correct"]]
        if incorrect:
            print("❌ INCORRECT PREDICTIONS:")
            for r in incorrect:
                print(f"   Text: {r['text'][:60]}...")
                print(f"   Expected: {r['expected']}, Got: {r['predicted']} (conf: {r['confidence']:.3f})")
                print()
        
        return results
        
    except Exception as e:
        print(f"❌ Error testing emotion classifier v2: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def test_legacy_emotion_classifier():
    """Test the legacy similarity-based emotion classifier."""
    print("🔄 Testing Legacy Emotion Classifier (Similarity)")
    print("=" * 60)
    
    try:
        from backend.app.services.emotion_classifier import classify_emotion as classify_emotion_legacy
        from backend.app.services.emotion_classifier import initialize_emotion_classifier
        
        # Initialize legacy classifier
        initialize_emotion_classifier()
        print("✅ Legacy classifier initialized")
        print()
        
        results = []
        total_time = 0
        
        for i, sample in enumerate(TEST_SAMPLES, 1):
            text = sample["text"]
            expected = sample["expected"]
            
            print(f"Test {i:2d}: {text[:50]}...")
            
            # Time the classification
            start_time = time.time()
            predicted_emotion = classify_emotion_legacy(text)
            end_time = time.time()
            
            classification_time = end_time - start_time
            total_time += classification_time
            
            # Check if prediction matches expected
            is_correct = predicted_emotion == expected
            
            print(f"         Expected: {expected}")
            print(f"         Predicted: {predicted_emotion}")
            print(f"         Time: {classification_time:.3f}s")
            print(f"         ✅ Correct" if is_correct else f"         ❌ Incorrect")
            print()
            
            results.append({
                "text": text,
                "expected": expected,
                "predicted": predicted_emotion,
                "time": classification_time,
                "correct": is_correct
            })
        
        # Calculate metrics
        correct_predictions = sum(1 for r in results if r["correct"])
        total_predictions = len(results)
        accuracy = correct_predictions / total_predictions
        avg_time = total_time / total_predictions
        
        print("📈 LEGACY PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.2%}")
        print(f"Average Time: {avg_time:.3f}s per classification")
        print(f"Total Time: {total_time:.3f}s")
        print()
        
        return results
        
    except Exception as e:
        print(f"❌ Error testing legacy emotion classifier: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def compare_classifiers():
    """Compare both emotion classifiers side by side."""
    print("🆚 COMPARING EMOTION CLASSIFIERS")
    print("=" * 60)
    
    # Test both classifiers
    enhanced_results = await test_emotion_classifier_v2()
    print("\n" + "=" * 60 + "\n")
    legacy_results = await test_legacy_emotion_classifier()
    
    if enhanced_results and legacy_results:
        print("\n" + "📊 COMPARISON SUMMARY" + "\n")
        print("=" * 60)
        
        # Calculate metrics for both
        enhanced_accuracy = sum(1 for r in enhanced_results if r["correct"]) / len(enhanced_results)
        legacy_accuracy = sum(1 for r in legacy_results if r["correct"]) / len(legacy_results)
        
        enhanced_avg_time = sum(r["time"] for r in enhanced_results) / len(enhanced_results)
        legacy_avg_time = sum(r["time"] for r in legacy_results) / len(legacy_results)
        
        enhanced_avg_conf = sum(r["confidence"] for r in enhanced_results) / len(enhanced_results)
        
        print(f"Enhanced Classifier (DistilBERT):")
        print(f"  ✅ Accuracy: {enhanced_accuracy:.2%}")
        print(f"  ⏱️  Avg Time: {enhanced_avg_time:.3f}s")
        print(f"  🔢 Avg Confidence: {enhanced_avg_conf:.3f}")
        print()
        
        print(f"Legacy Classifier (Similarity):")
        print(f"  ✅ Accuracy: {legacy_accuracy:.2%}")
        print(f"  ⏱️  Avg Time: {legacy_avg_time:.3f}s")
        print(f"  🔢 Confidence: N/A")
        print()
        
        # Performance comparison
        accuracy_improvement = enhanced_accuracy - legacy_accuracy
        time_ratio = enhanced_avg_time / legacy_avg_time
        
        print(f"📈 IMPROVEMENTS:")
        print(f"  Accuracy: {accuracy_improvement:+.1%}")
        print(f"  Speed: {time_ratio:.1f}x {'slower' if time_ratio > 1 else 'faster'}")
        print()
        
        # Side-by-side comparison for interesting cases
        print("🔍 DETAILED COMPARISON (first 5 samples):")
        for i in range(min(5, len(TEST_SAMPLES))):
            sample = TEST_SAMPLES[i]
            enhanced = enhanced_results[i]
            legacy = legacy_results[i]
            
            print(f"\nSample {i+1}: {sample['text'][:40]}...")
            print(f"  Expected: {sample['expected']}")
            print(f"  Enhanced: {enhanced['predicted']} (conf: {enhanced['confidence']:.3f}) {'✅' if enhanced['correct'] else '❌'}")
            print(f"  Legacy:   {legacy['predicted']} {'✅' if legacy['correct'] else '❌'}")

async def main():
    """Main test function."""
    print("🧪 EMOTION CLASSIFIER TESTING SUITE")
    print("=" * 60)
    print()
    
    await compare_classifiers()

if __name__ == "__main__":
    asyncio.run(main())