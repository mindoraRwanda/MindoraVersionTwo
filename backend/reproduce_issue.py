
import asyncio
from app.services.emotion_classifier import LLMEmotionClassifier

async def reproduce_type_error():
    classifier = LLMEmotionClassifier()
    
    # Simulate a malformed LLM response where keywords is a string instead of a list
    malformed_response = """
    {
        "selected_emotion": "sadness",
        "confidence": 0.9,
        "keywords": "sadness, grief", 
        "cultural_emotional_indicators": ["indirect expression"],
        "youth_emotional_patterns": ["withdrawal"],
        "secondary_emotions": ["loneliness"]
    }
    """
    
    print("Attempting to parse malformed response...")
    try:
        result = classifier._parse_emotion_response(malformed_response, "test text", "en")
        print("Result:", result)
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(reproduce_type_error())
