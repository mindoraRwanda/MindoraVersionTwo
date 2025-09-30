# Model Setup Guide - Avoiding Server Hanging Issues

## Problem
The HuggingFace SmolLM3 model can cause your server to hang during startup because it downloads the model (~1.5GB) when first used. This creates long startup times and potential timeouts.

## Solution
Use the provided scripts to pre-download models before starting the server.

## Quick Start

### 1. Download Models First
```bash
# Download all required models (recommended)
python -m backend.app.utils.download_models

# Or use the shell script
./cache_models.sh
```

### 2. Start Server with Alternative Provider
```bash
# Use Ollama (recommended for development)
python setup_server.py --provider ollama

# Use OpenAI (requires API key)
export OPENAI_API_KEY="your-key-here"
python setup_server.py --provider openai

# Use Groq (requires API key)
export GROQ_API_KEY="your-key-here"
python setup_server.py --provider groq
```

### 3. Or Use HuggingFace After Pre-downloading
```bash
# After running the download script
python setup_server.py --provider huggingface
```

## Available Scripts

### `download_models.py`
Downloads all required models:
- **SmolLM3-3B** (~1.5GB) - Main chatbot model
- **BAAI/bge-small-en** (~100MB) - Emotion classification
- **all-MiniLM-L6-v2** (~80MB) - RAG embeddings

**Usage:**
```bash
# Download all models
python -m backend.app.utils.download_models

# Test if models are already downloaded
python -m backend.app.utils.download_models --test-only

# Skip specific models
python -m backend.app.utils.download_models --skip-emotion --skip-rag

# Create shell script for automation
python -m backend.app.utils.download_models --create-script
```

### `setup_server.py`
Helps configure and start the server with proper provider selection.

**Usage:**
```bash
# Interactive setup
python setup_server.py

# Start with specific provider
python setup_server.py --provider ollama

# Test configuration only
python setup_server.py --test-only

# Create startup scripts for all providers
python setup_server.py --create-script
```

## Provider Recommendations

### 🥇 Ollama (Recommended for Development)
- **Pros:** No API keys needed, local models, fast inference
- **Cons:** Requires Ollama server running
- **Setup:**
  ```bash
  # Install Ollama
  curl -fsSL https://ollama.ai/install.sh | sh

  # Start Ollama service
  ollama serve

  # Pull a model (in another terminal)
  ollama pull llama2
  ```

### 🥈 OpenAI
- **Pros:** High quality, reliable
- **Cons:** Requires API key, costs money
- **Setup:**
  ```bash
  export OPENAI_API_KEY="your-key-here"
  ```

### 🥉 Groq
- **Pros:** Fast, free tier available
- **Cons:** Requires API key
- **Setup:**
  ```bash
  export GROQ_API_KEY="your-key-here"
  ```

### ⚠️ HuggingFace (Local Models)
- **Pros:** Completely local, no API keys
- **Cons:** Slow startup, high memory usage, may hang
- **Setup:** Run `python -m backend.app.utils.download_models` first

## Troubleshooting

### Server Still Hanging?
1. **Check if models are downloaded:**
   ```bash
   python -m backend.app.utils.download_models --test-only
   ```

2. **Use Ollama instead:**
   ```bash
   python setup_server.py --provider ollama
   ```

3. **Check available providers:**
   ```bash
   python setup_server.py --test-only
   ```

### Out of Disk Space?
- Models require ~2GB total space
- Check space: `df -h`
- Clear cache: `rm -rf ~/.cache/huggingface/hub`

### Network Issues During Download?
- Downloads may timeout on slow connections
- Try again later or use a faster connection
- Models are cached after first download

## Environment Variables

```bash
# LLM Provider Selection
export OPENAI_API_KEY="sk-..."          # Use OpenAI
export GROQ_API_KEY="gsk_..."           # Use Groq
export OLLAMA_BASE_URL="http://localhost:11434"  # Use Ollama

# HuggingFace Configuration
export DEFAULT_MODEL="HuggingFaceTB/SmolLM3-3B"
export HUGGINGFACE_PRELOAD_MODEL="false"  # Set to "true" to preload models

# Server Configuration
export LLM_PROVIDER="ollama"            # Default provider
```

## Startup Scripts

The setup script can create individual startup scripts:

```bash
# Create startup scripts for all providers
python setup_server.py --create-script

# This creates:
# - start_ollama.sh
# - start_openai.sh
# - start_groq.sh
# - start_huggingface.sh
```

## Performance Tips

1. **Use Ollama for development** - fastest startup
2. **Pre-download models** - avoid runtime downloads
3. **Use appropriate model sizes** - smaller models = faster startup
4. **Monitor memory usage** - HuggingFace models use significant RAM

## Getting Help

If you continue to have issues:

1. Check the troubleshooting section above
2. Verify your internet connection for downloads
3. Ensure sufficient disk space (~2GB free)
4. Try using Ollama instead of HuggingFace
5. Check the logs for specific error messages

The scripts are designed to be robust and provide clear error messages to help diagnose issues.