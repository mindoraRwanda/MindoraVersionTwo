
import json
import re
import sys
from unittest.mock import MagicMock

# Mock heavy dependencies
sys.modules["llama_index"] = MagicMock()
sys.modules["llama_index.core"] = MagicMock()
sys.modules["llama_index.vector_stores"] = MagicMock()
sys.modules["llama_index.vector_stores.qdrant"] = MagicMock()
sys.modules["llama_index.embeddings"] = MagicMock()
sys.modules["llama_index.embeddings.huggingface"] = MagicMock()
sys.modules["llama_index.embeddings.ollama"] = MagicMock()
sys.modules["llama_index.llms"] = MagicMock()
sys.modules["llama_index.llms.ollama"] = MagicMock()
sys.modules["llama_index.llms.openai"] = MagicMock()
sys.modules["llama_index.llms.groq"] = MagicMock()
sys.modules["qdrant_client"] = MagicMock()
sys.modules["qdrant_client.models"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["fitz"] = MagicMock()

from app.services.emotion_classifier import LLMEmotionClassifier

def reproduce_json_error():
    classifier = LLMEmotionClassifier()
    
    # Simulate a malformed LLM response that might cause "Expecting ',' delimiter"
    # This often happens with unescaped quotes or missing commas
    malformed_response = """
    {
        "selected_emotion": "anger"
        "confidence": 0.9,
        "reasoning": "The user is angry about "something" specific",
        "keywords": ["anger", "furious"]
    }
    """
    
    print("Attempting to parse malformed JSON...")
    result = classifier._parse_emotion_response(malformed_response, "test text", "en")
    print("Result:", result)
    
    if result.get("confidence") == 0.3 and result.get("emotion") == "neutral":
        print("FAIL: Fallback triggered, parsing failed.")
    else:
        print("SUCCESS: Parsed successfully (or handled gracefully).")

if __name__ == "__main__":
    reproduce_json_error()
