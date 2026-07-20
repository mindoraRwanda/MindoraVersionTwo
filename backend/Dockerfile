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

# Upgrade pip and install Python dependencies
# This layer is cached unless requirements.txt changes
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code last
# This layer rebuilds on every code change, but pip install is cached
COPY . .

EXPOSE 8000

# Shell form (not exec/JSON-array form) so $PORT actually expands. Railway
# injects its own PORT env var at runtime; ${PORT:-8000} falls back to 8000
# for local/docker-compose use where PORT isn't set.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
