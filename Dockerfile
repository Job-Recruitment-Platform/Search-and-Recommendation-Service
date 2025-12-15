# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Optimize pip
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Upgrade pip and setuptools
RUN pip install --upgrade pip "setuptools<81" wheel

# Install PyTorch CPU (required by embeddings)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install milvus-model explicitly (sometimes missed by requirements.txt)
RUN pip install milvus-model>=0.2.0

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create cache directories
RUN mkdir -p /home/appuser/.cache/huggingface/hub && \
    chown -R appuser:appuser /home/appuser/.cache

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # Flask config
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=8000 \
    FLASK_DEBUG=false \
    \
    # Milvus config
    MILVUS_HOST=milvus \
    MILVUS_PORT=19530 \
    \
    # Redis config
    REDIS_HOST=redis \
    REDIS_PORT=6379 \
    REDIS_DB=0 \
    \
    # Stream config (Outbox events for Job sync)
    OUTBOX_STREAM_NAME=outbox-events \
    OUTBOX_CONSUMER_GROUP=outbox-processor-group \
    OUTBOX_CONSUMER_NAME=python-sync-worker-1 \
    \
    # Stream config (User interactions for recommendations)
    INTERACTION_STREAM_NAME=user-interactions \
    INTERACTION_CONSUMER_GROUP=recommend-service-group \
    INTERACTION_CONSUMER_NAME=python-recommend-worker-1 \
    \
    # Embedding model config
    EMBEDDING_MODEL_NAME=BAAI/bge-m3 \
    EMBEDDING_DEVICE=cpu \
    EMBEDDING_USE_FP16=false \
    \
    # Search config
    SEARCH_DEFAULT_LIMIT=10 \
    SEARCH_DEFAULT_OFFSET=0 \
    SEARCH_THRESHOLD=0.3 \
    \
    # Recommendation config
    CANDIDATE_API_BASE_URL=http://backend:8080 \
    INTERACTION_HALF_LIFE_DAYS=30 \
    CF_MODEL_PATH=/app/CFModel/models/cf_model.pkl \
    \
    # Hugging Face cache
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/hub

# Copy application code
COPY --chown=appuser:appuser . .

# Create data and model directories
RUN mkdir -p /app/data /app/CFModel/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=360s --retries=10 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run Flask app with 2 consumer threads
CMD ["python", "main.py"]