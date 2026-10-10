# TaxonVision-MLOps Production Serving Container
# Optimized build using Pixi and non-root execution

FROM ghcr.io/prefix-dev/pixi:latest AS runner

WORKDIR /app

# Ensure non-root app user exists and owns working directory
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser && \
    chown -R appuser:appgroup /app

USER appuser

# Copy dependency specifications, README, and source code required for editable hatchling build
COPY --chown=appuser:appgroup pyproject.toml pixi.lock README.md ./
COPY --chown=appuser:appgroup src/ src/

# Install the minimal CPU-only production/inference environment directly as non-root user
RUN pixi install --frozen -e ci

# Copy configuration, model checkpoints, and default parameters
COPY --chown=appuser:appgroup config/ config/
COPY --chown=appuser:appgroup models/checkpoints/head.pt models/checkpoints/head.pt

# Expose FastAPI HTTP serving port
EXPOSE 8000

ENV PORT=8000 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1

# Healthcheck hitting the readiness probe (validating model warmup & cache)
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD pixi run --frozen -e ci python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')" || exit 1

ENTRYPOINT ["pixi", "run", "--frozen", "-e", "ci", "uvicorn", "taxon_vision.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
