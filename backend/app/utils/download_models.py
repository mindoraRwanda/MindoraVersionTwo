#!/usr/bin/env python3
"""
Model Download Script for Mindora Therapy Chatbot

This script pre-downloads required models to avoid hanging during server startup.
It downloads the SmolLM3 model and other transformer models needed for the chatbot.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import subprocess

def check_disk_space():
    """Check available disk space."""
    try:
        stat = os.statvfs('.')
        available_gb = (stat.f_available * stat.f_frsize) / (1024**3)
        print(f"💾 Available disk space: {available_gb:.1f} GB")
        return available_gb > 5  # Require at least 5GB free
    except:
        print("⚠️  Could not check disk space")
        return True

def download_smollm3_model():
    """Download the SmolLM3 model."""
    print("\n🤖 Downloading SmolLM3 model...")
    print("This model is ~1.5GB and may take several minutes to download.")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_path = "HuggingFaceTB/SmolLM3-3B"

        print(f"📥 Downloading tokenizer for {model_path}...")
        start_time = time.time()

        # Download tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            timeout=300  # 5 minute timeout
        )

        tokenizer_time = time.time() - start_time
        print(f"✅ Tokenizer downloaded in {tokenizer_time:.1f} seconds")

        print(f"📥 Downloading model {model_path}...")
        start_time = time.time()

        # Download model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto"
        )

        model_time = time.time() - start_time
        print(f"✅ Model downloaded in {model_time:.1f} seconds")

        # Save to cache directory
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        print(f"💾 Model cached at: {cache_dir}")

        return True

    except Exception as e:
        print(f"❌ Failed to download SmolLM3 model: {e}")
        return False

def download_emotion_model():
    """Download the emotion classification model."""
    print("\n🎭 Downloading emotion classification model...")

    try:
        from sentence_transformers import SentenceTransformer

        model_name = "BAAI/bge-small-en"

        print(f"📥 Downloading {model_name}...")
        start_time = time.time()

        model = SentenceTransformer(model_name)

        download_time = time.time() - start_time
        print(f"✅ Emotion model downloaded in {download_time:.1f} seconds")

        return True

    except Exception as e:
        print(f"❌ Failed to download emotion model: {e}")
        return False

def download_rag_model():
    """Download the RAG embedding model."""
    print("\n🔍 Downloading RAG embedding model...")

    try:
        from sentence_transformers import SentenceTransformer

        model_name = "all-MiniLM-L6-v2"

        print(f"📥 Downloading {model_name}...")
        start_time = time.time()

        model = SentenceTransformer(model_name)

        download_time = time.time() - start_time
        print(f"✅ RAG model downloaded in {download_time:.1f} seconds")

        return True

    except Exception as e:
        print(f"❌ Failed to download RAG model: {e}")
        return False

def test_model_loading():
    """Test that models can be loaded successfully."""
    print("\n🧪 Testing model loading...")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from sentence_transformers import SentenceTransformer

        # Test SmolLM3
        print("Testing SmolLM3 model...")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM3-3B", dtype="auto")
        print("✅ SmolLM3 model loads successfully")

        # Test emotion model
        print("Testing emotion model...")
        emotion_model = SentenceTransformer("BAAI/bge-small-en")
        print("✅ Emotion model loads successfully")

        # Test RAG model
        print("Testing RAG model...")
        rag_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ RAG model loads successfully")

        return True

    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def create_model_cache_script():
    """Create a script to cache models."""
    script_content = '''#!/bin/bash
# Model Caching Script
# This script pre-downloads models to avoid hanging during server startup

echo "🚀 Starting model download process..."
echo "This may take 10-30 minutes depending on your internet connection."

# Create models directory if it doesn't exist
mkdir -p ~/.cache/huggingface/hub

# Run the model download script
python -m backend.app.utils.download_models

if [ $? -eq 0 ]; then
    echo "✅ All models downloaded successfully!"
    echo "💡 You can now start the server without hanging issues."
else
    echo "❌ Some models failed to download."
    echo "💡 Check the error messages above and try again."
fi
'''

    script_path = "cache_models.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    print(f"✅ Created model caching script: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Model Download Script")
    parser.add_argument("--test-only", action="store_true",
                       help="Only test if models are already downloaded")
    parser.add_argument("--skip-smollm3", action="store_true",
                       help="Skip downloading SmolLM3 model")
    parser.add_argument("--skip-emotion", action="store_true",
                       help="Skip downloading emotion model")
    parser.add_argument("--skip-rag", action="store_true",
                       help="Skip downloading RAG model")
    parser.add_argument("--create-script", action="store_true",
                       help="Create a shell script for model caching")

    args = parser.parse_args()

    print("🌟 Mindora Model Download Script")
    print("=" * 40)

    # Check disk space
    if not check_disk_space():
        print("❌ Not enough disk space. Please free up at least 5GB.")
        return 1

    if args.create_script:
        create_model_cache_script()
        return 0

    if args.test_only:
        success = test_model_loading()
        return 0 if success else 1

    print("📋 This script will download the following models:")
    print("  🤖 SmolLM3-3B (~1.5GB) - Main chatbot model")
    print("  🎭 BAAI/bge-small-en (~100MB) - Emotion classification")
    print("  🔍 all-MiniLM-L6-v2 (~80MB) - RAG embeddings")

    total_size = "~1.7GB"
    print(f"\n💾 Total download size: {total_size}")

    confirm = input("\nProceed with downloads? (y/N): ").lower().strip()
    if confirm not in ['y', 'yes']:
        print("❌ Download cancelled")
        return 1

    # Download models
    success_count = 0
    total_count = 0

    if not args.skip_smollm3:
        total_count += 1
        if download_smollm3_model():
            success_count += 1

    if not args.skip_emotion:
        total_count += 1
        if download_emotion_model():
            success_count += 1

    if not args.skip_rag:
        total_count += 1
        if download_rag_model():
            success_count += 1

    # Test model loading
    print(f"\n📊 Download Results: {success_count}/{total_count} models downloaded successfully")

    if success_count == total_count:
        print("✅ All models downloaded successfully!")
        print("💡 You can now start the server without hanging issues.")

        # Test loading
        if test_model_loading():
            print("🎉 All models load successfully!")
        else:
            print("⚠️  Models downloaded but loading test failed.")
            print("💡 This might be normal - try starting the server.")
    else:
        print("❌ Some models failed to download.")
        print("💡 Check the error messages above and try again.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())