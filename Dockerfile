# Multi-stage build - Optimized to reduce layer size

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Install minimal build dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment to disable hash checking
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU FIRST (required by BGE-M3)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
RUN pip install \
    Flask==3.0.0 \
    redis==5.0.1 \
    numpy==1.26.2 \
    python-dotenv==1.0.0 \
    scipy==1.11.4 \
    requests==2.31.0

# Install pymilvus dependencies
RUN pip install \
    marshmallow==3.20.1 \
    marshmallow-enum==1.5.1 \
    environs==9.5.0 \
    grpcio==1.60.0 \
    protobuf==3.20.0

# Install pymilvus
RUN pip install pymilvus==2.4.0

# Install missing dependencies for pymilvus[model]
RUN pip install \
    transformers \
    sentence-transformers \
    datasets \
    huggingface-hub \
    tokenizers

# Install pymilvus[model] - Includes BGE-M3 dependencies
RUN pip install "pymilvus[model]==2.4.0"

# Install FlagEmbedding explicitly
RUN pip install FlagEmbedding

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create user FIRST (before copying files)
RUN useradd -m -u 1000 appuser

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Create cache directories with correct ownership
RUN mkdir -p /home/appuser/.cache/huggingface/hub && \
    chown -R appuser:appuser /home/appuser/.cache

# Set environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=8000 \
    FLASK_DEBUG=false \
    MILVUS_HOST=milvus \
    MILVUS_PORT=19530 \
    REDIS_HOST=redis \
    REDIS_PORT=6379 \
    REDIS_DB=0 \
    REDIS_STREAM_NAME=outbox-events \
    REDIS_CONSUMER_GROUP=milvus-sync \
    REDIS_CONSUMER_NAME=worker-1 \
    INTERACTION_STREAM_NAME=outbox-events \
    INTERACTION_CONSUMER_GROUP=recommend-service-group \
    INTERACTION_HALF_LIFE_DAYS=30 \
    EMBEDDING_MODEL_NAME=BAAI/bge-m3 \
    EMBEDDING_DEVICE=cpu \
    EMBEDDING_USE_FP16=false \
    SEARCH_DEFAULT_LIMIT=10 \
    SEARCH_DEFAULT_OFFSET=0 \
    SEARCH_THRESHOLD=0.3 \
    CANDIDATE_API_BASE_URL=http://backend:8080 \
    CF_MODEL_PATH=/app/CFModel/models/cf_model.pkl \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/hub

# Copy application code
COPY --chown=appuser:appuser . .

# Create app directories with correct ownership
RUN mkdir -p /app/data /app/CFModel/models && \
    chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=360s --retries=10 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "main.py"]