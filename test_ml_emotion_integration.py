"""
Quick test script to verify ML emotion classifier integration into pipeline.
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

async def test_ml_emotion_classifier():
    """Test that ML emotion classifier initializes and works."""
    print("🧪 Testing ML Emotion Classifier Integration\n")
    
    # Test 1: Import the classifier
    print("1️⃣ Testing import...")
    try:
        from backend.app.services.emotion.text_emotion_classifier import TextEmotionClassifier
        print("   ✅ Import successful")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return
    
    # Test 2: Initialize classifier
    print("\n2️⃣ Testing initialization...")
    try:
        classifier = TextEmotionClassifier()
        print("   ✅ Classifier initialized")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return
    
    # Test 3: Classify emotion
    print("\n3️⃣ Testing classification...")
    test_texts = [
        "I feel so sad and alone. Nobody understands me.",
        "I'm really angry at how unfair this is!",
        "I'm so happy and excited about the future!",
        "Ndi mfite ikibazo, I don't know what to do anymore."  # Kinyarwanda marker
    ]
    
    for text in test_texts:
        try:
            result = await classifier.classify(text, context={})
            print(f"\n   Text: '{text[:50]}...'")
            print(f"   ✅ Emotion: {result.primary_emotion.value}")
            print(f"   ✅ Intensity: {result.intensity.value}")
            print(f"   ✅ Confidence: {result.confidence:.2f}")
            if result.cultural_context:
                print(f"   🌍 Cultural: {result.cultural_context}")
        except Exception as e:
            print(f"   ❌ Classification failed: {e}")
    
    # Test 4: Test pipeline integration
    print("\n\n4️⃣ Testing pipeline integration...")
    try:
        from backend.app.services.stateful_pipeline import StatefulMentalHealthPipeline
        print("   ✅ Pipeline import successful")
        
        # Initialize pipeline (this will initialize ML classifier internally)
        print("   🔧 Initializing pipeline...")
        pipeline = StatefulMentalHealthPipeline(llm_provider=None, rag_service=None)
        
        if pipeline.ml_emotion_classifier:
            print("   ✅ ML classifier integrated into pipeline!")
        else:
            print("   ⚠️ ML classifier not initialized in pipeline")
            
    except Exception as e:
        print(f"   ❌ Pipeline integration failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\n✅ Integration test complete!")

if __name__ == "__main__":
    asyncio.run(test_ml_emotion_classifier())
