# ===========================================
# PyHellen API - CPU Version
# Multi-stage build with uv for fast installs
# ===========================================

# Stage 1: Builder - Install dependencies with uv
FROM python:3.10-slim AS builder

WORKDIR /build

# Install uv (ultra-fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment with uv
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with uv (much faster than pip)
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt


# Stage 2: Runtime - Minimal production image
FROM python:3.10-slim AS runtime

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIE_EXTENDED_DOWNLOADS=/data/models \
    USE_CUDA=0 \
    DOCKER_CONTAINER=true \
    LOG_FORMAT=json \
    LOG_LEVEL=INFO

# Create model directory
RUN mkdir -p /data/models

# Copy application code
COPY app/ ./app/

# Create a non-root user and switch to it
RUN useradd -m -u 1000 pyhellen && \
    chown -R pyhellen:pyhellen /app /data
USER pyhellen

# Expose the port
EXPOSE 8000

# Health check - verify the API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/service/health || exit 1

# Run the application with production settings
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
