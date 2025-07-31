# Multi-stage build for production - optimized for size
FROM python:3.11-slim AS builder

# Install build dependencies with retry and timeout
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Runtime stage - smaller final image
FROM python:3.11-slim

# Install only runtime dependencies with retry
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy only necessary application files
COPY --chown=appuser:appuser ./api ./api
COPY --chown=appuser:appuser ./models ./models
COPY --chown=appuser:appuser ./services ./services
COPY --chown=appuser:appuser ./core ./core
COPY --chown=appuser:appuser ./data ./data
COPY --chown=appuser:appuser ./tasks ./tasks
COPY --chown=appuser:appuser ./ml/fantasy_predictor.py ./ml/fantasy_predictor.py
COPY --chown=appuser:appuser ./ml/neural_network.py ./ml/neural_network.py
COPY --chown=appuser:appuser ./ml/feature_engineer.py ./ml/feature_engineer.py
COPY --chown=appuser:appuser ./ml/draft_optimizer.py ./ml/draft_optimizer.py
COPY --chown=appuser:appuser ./ml/__init__.py ./ml/__init__.py
COPY --chown=appuser:appuser ./main.py ./main.py
COPY --chown=appuser:appuser ./celery_app.py ./celery_app.py
COPY --chown=appuser:appuser ./__init__.py ./__init__.py
COPY --chown=appuser:appuser ./alembic.ini ./alembic.ini
COPY --chown=appuser:appuser ./alembic ./alembic

# Download pre-trained models (if not mounted)
# In production, you'd mount these from external storage
RUN mkdir -p /app/saved_models && chown -R appuser:appuser /app/saved_models

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run with optimized settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--loop", "uvloop"]