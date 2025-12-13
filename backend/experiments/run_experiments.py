import asyncio
import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env.development"))
load_dotenv(env_path)

# Add backend directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock heavy dependencies that are not needed for emotion classification
# This avoids dependency hell with llama-index, pydantic, etc.
from unittest.mock import MagicMock
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

from app.services.emotion_classifier import initialize_emotion_classifier, classify_emotion

from app.services.llm_providers import create_llm_provider

async def run_experiments():
    print("Starting experiments...")
    
    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} examples from {dataset_path}")
    
    # Initialize classifier with Real Provider
    # User requested Ollama
    try:
        real_provider = create_llm_provider(provider="ollama", model="granite4:1b-h")
        print(f"Initialized LLM Provider: {real_provider.provider_name} (Model: {real_provider.model_name})")
    except Exception as e:
        print(f"Failed to initialize LLM Provider: {e}")
        return

    classifier = initialize_emotion_classifier(real_provider)
    
    results = []
    
    for i, example in enumerate(dataset):
        text = example["text"]
        true_label = example["label"]
        
        print(f"[{i+1}/{len(dataset)}] Classifying: {text[:30]}...")
        
        try:
            # Use the instance method directly
            prediction_result = await classifier.classify_emotion(text)
            
            predicted_label = prediction_result.get("emotion", "neutral")
            confidence = prediction_result.get("confidence", 0.0)
            emotions_dist = prediction_result.get("emotions", {})
            
            result_entry = {
                "text": text,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "emotions_distribution": emotions_dist,
                "full_response": prediction_result
            }
            
            results.append(result_entry)
            
        except Exception as e:
            print(f"Error classifying index {i}: {e}")
            results.append({
                "text": text,
                "true_label": true_label,
                "predicted_label": "error",
                "error": str(e)
            })
            
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Experiments completed. Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(run_experiments())
