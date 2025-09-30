#!/usr/bin/env python3
"""
Server Setup Script for Mindora Therapy Chatbot

This script helps configure and start the server with proper model loading.
It provides options to avoid hanging issues during startup.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def check_environment():
    """Check if required environment variables are set."""
    print("🔍 Checking environment configuration...")

    # Check for required environment variables
    required_vars = [
        ("OPENAI_API_KEY", "OpenAI API key for GPT models"),
        ("GROQ_API_KEY", "Groq API key for fast inference"),
        ("OLLAMA_BASE_URL", "Ollama server URL (default: http://127.0.0.1:11434)")
    ]

    configured_providers = []

    for var, description in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {description}")
            if var == "OPENAI_API_KEY":
                configured_providers.append("openai")
            elif var == "GROQ_API_KEY":
                configured_providers.append("groq")
            elif var == "OLLAMA_BASE_URL":
                configured_providers.append("ollama")
        else:
            print(f"⚠️  {var}: Not configured")

    # Check for HuggingFace (local models)
    hf_model = os.getenv("DEFAULT_MODEL", "HuggingFaceTB/SmolLM3-3B")
    print(f"🤖 HuggingFace model: {hf_model}")
    configured_providers.append("huggingface")

    return configured_providers

def test_provider_availability():
    """Test which providers are available."""
    print("\n🔍 Testing provider availability...")

    try:
        from backend.app.services.llm_providers import LLMProviderFactory

        available_providers = LLMProviderFactory.get_available_providers()

        for provider, available in available_providers.items():
            status = "✅ Available" if available else "❌ Unavailable"
            print(f"  {provider}: {status}")

        return available_providers

    except ImportError as e:
        print(f"❌ Error importing providers: {e}")
        return {}

def start_server_with_provider(provider_name, host="0.0.0.0", port=8000):
    """Start the server with a specific provider."""
    print(f"\n🚀 Starting server with {provider_name} provider...")

    # Set environment variable for the provider
    env = os.environ.copy()
    env["LLM_PROVIDER"] = provider_name

    # Start the server
    try:
        cmd = [
            sys.executable, "-m", "uvicorn",
            "backend.app.main:app",
            "--host", host,
            "--port", str(port),
            "--reload"
        ]

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, env=env, check=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True

    return True

def create_startup_script(provider_name):
    """Create a startup script for the specified provider."""
    script_content = f'''#!/bin/bash
# Auto-generated startup script for {provider_name} provider

export LLM_PROVIDER={provider_name}

# Check if Ollama is running (for ollama provider)
if [ "$LLM_PROVIDER" = "ollama" ]; then
    echo "🔍 Checking Ollama service..."
    if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
        echo "❌ Ollama service not running. Please start Ollama first:"
        echo "   ollama serve"
        exit 1
    fi
    echo "✅ Ollama service is running"
fi

# Start the server
echo "🚀 Starting Mindora server with {provider_name} provider..."
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
'''

    script_path = f"start_{provider_name}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make executable
    os.chmod(script_path, 0o755)
    print(f"✅ Created startup script: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Mindora Server Setup Script")
    parser.add_argument("--provider", "-p",
                       choices=["ollama", "openai", "groq", "huggingface"],
                       help="LLM provider to use")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--create-script", action="store_true",
                       help="Create startup scripts for all providers")
    parser.add_argument("--test-only", action="store_true",
                       help="Only test configuration, don't start server")

    args = parser.parse_args()

    print("🌟 Mindora Therapy Chatbot Server Setup")
    print("=" * 50)

    # Check environment
    configured_providers = check_environment()

    # Test provider availability
    available_providers = test_provider_availability()

    # Determine provider to use
    if args.provider:
        provider_name = args.provider
    elif configured_providers:
        # Use first available configured provider
        for provider in ["openai", "groq", "ollama", "huggingface"]:
            if provider in configured_providers and available_providers.get(provider, False):
                provider_name = provider
                break
        else:
            provider_name = "ollama"  # fallback
    else:
        provider_name = "ollama"  # default fallback

    print(f"\n🎯 Selected provider: {provider_name}")

    if args.create_script:
        print("\n📝 Creating startup scripts...")
        for provider in ["ollama", "openai", "groq", "huggingface"]:
            create_startup_script(provider)
        return

    if args.test_only:
        print("\n✅ Configuration test completed")
        return

    # Start server
    print(f"\n📋 Provider recommendations:")
    print(f"  • Use 'ollama' for local models (requires Ollama server)")
    print(f"  • Use 'openai' for GPT models (requires API key)")
    print(f"  • Use 'groq' for fast inference (requires API key)")
    print(f"  • Use 'huggingface' for local models (may hang on startup)")

    confirm = input(f"\nStart server with '{provider_name}' provider? (y/N): ").lower().strip()
    if confirm in ['y', 'yes']:
        success = start_server_with_provider(provider_name, args.host, args.port)
        if success:
            print("✅ Server started successfully!")
        else:
            print("❌ Server failed to start. Check the error messages above.")
            return 1
    else:
        print("❌ Server startup cancelled")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())