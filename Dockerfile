# syntax=docker/dockerfile:1
# TaxonVision-MLOps Production Serving Container
# Optimized multi-stage build using Pixi and non-root execution

FROM ghcr.io/prefix-dev/pixi:latest AS runner

WORKDIR /app

# Ensure non-root app user exists
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy dependency specifications first to leverage layer caching
COPY pyproject.toml pixi.lock ./

# Install the minimal CPU-only production/inference environment
RUN pixi install --frozen -e ci

# Copy application source code, configuration, and default parameters
COPY src/ src/
COPY config/ config/
COPY models/checkpoints/head.pt models/checkpoints/head.pt

# Ensure files are owned by the non-root application user
RUN chown -R appuser:appgroup /app

USER appuser

# Expose FastAPI HTTP serving port
EXPOSE 8000

ENV PORT=8000 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1

# Healthcheck hitting the readiness probe (validating model warmup & cache)
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD pixi run --frozen -e ci python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')" || exit 1

ENTRYPOINT ["pixi", "run", "--frozen", "-e", "ci", "uvicorn", "taxon_vision.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
