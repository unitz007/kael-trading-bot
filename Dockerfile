# =============================================================================
# Kael Trading Bot — Backend Dockerfile
# =============================================================================
# Multi-stage build: installs Python dependencies in a virtualenv during
# the "builder" stage, then copies only the installed packages into the
# lean production image.
# =============================================================================

# --- Build stage ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (required for some Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Create a virtualenv for dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install the application package itself
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps -e .

# --- Production stage ---
FROM python:3.11-slim AS production

WORKDIR /app

# Copy the virtualenv from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source code
COPY src/ src/
COPY pyproject.toml .

# Create directories used at runtime
RUN mkdir -p models logs .cache/forex_data

# Environment variables with sensible defaults (override via docker-compose or -e)
ENV KAEL_HOST=0.0.0.0 \
    KAEL_PORT=5000 \
    KAEL_START_DATE=2020-01-01 \
    KAEL_END_DATE=2025-01-01 \
    KAEL_INTERVAL=1d \
    KAEL_CACHE_DIR=.cache/forex_data \
    PYTHONUNBUFFERED=1

EXPOSE ${KAEL_PORT}

# Health check — probe the /api/v1/pairs endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${KAEL_PORT}/api/v1/pairs')" || exit 1

# Start the API server
CMD ["sh", "-c", "uvicorn kael_trading_bot.api:create_app --factory --host ${KAEL_HOST} --port ${KAEL_PORT}"]
