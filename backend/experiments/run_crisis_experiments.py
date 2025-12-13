import asyncio
import json
import os
import sys
from unittest.mock import MagicMock
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env.development"))
load_dotenv(env_path)

# Add backend directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock heavy dependencies that are not needed for crisis classification
# This avoids dependency hell with llama-index, pydantic, etc.
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

# Import the crisis classifier
# We import directly to avoid triggering too many other imports, but the mocks should protect us
from app.services.crisis_classifier import classify_crisis

async def run_crisis_experiments():
    print("Starting crisis detection experiments...")
    
    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "crisis_dataset.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} examples from {dataset_path}")
    
    results = []
    
    for i, example in enumerate(dataset):
        text = example["text"]
        true_label = example["label"]
        
        print(f"[{i+1}/{len(dataset)}] Classifying: {text[:30]}...")
        
        try:
            # classify_crisis is synchronous in the file I viewed, but let's check if it needs await
            # The file content showed: def classify_crisis(text: str) -> CrisisResult:
            # It uses client.chat.completions.create which is sync.
            # So we call it synchronously.
            
            prediction_result = await classify_crisis(text)
            
            predicted_label = prediction_result.get("label", "other")
            severity = prediction_result.get("severity", "low")
            confidence = prediction_result.get("confidence", 0.0)
            rationale = prediction_result.get("rationale", "")
            
            result_entry = {
                "text": text,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "severity": severity,
                "confidence": confidence,
                "rationale": rationale,
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
    output_path = os.path.join(os.path.dirname(__file__), "crisis_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Experiments completed. Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(run_crisis_experiments())
