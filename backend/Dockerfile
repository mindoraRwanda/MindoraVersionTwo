FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    cmake \
    make \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
# This layer will only rebuild if requirements.txt changes
COPY requirements.txt .

# Upgrade build tooling, then install CPU-only PyTorch. The production
# Railway service has no CUDA GPU; the application already falls back to CPU.
# Pinning the CPU wheels avoids pulling NVIDIA runtime libraries while keeping
# SentenceTransformers and Transformers available for RAG and emotion analysis.
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       --extra-index-url https://pypi.org/simple \
       torch==2.3.0+cpu torchvision==0.18.0+cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code last
# This layer rebuilds on every code change, but pip install is cached
COPY . .

EXPOSE 8000

# Shell form (not exec/JSON-array form) so $PORT actually expands. Railway
# injects its own PORT env var at runtime; ${PORT:-8000} falls back to 8000
# for local/docker-compose use where PORT isn't set.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
