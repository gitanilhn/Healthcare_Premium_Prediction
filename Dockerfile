# ============================================================
# Stage 1: Build Python dependencies
# ============================================================

FROM python:3.10-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# System dependency required by XGBoost
# ------------------------------------------------------------

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Install uv
# ------------------------------------------------------------

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

# ------------------------------------------------------------
# Copy dependency definitions
# ------------------------------------------------------------

COPY pyproject.toml uv.lock ./

# ------------------------------------------------------------
# Create production virtual environment
# Uses EXACT versions from uv.lock
# ------------------------------------------------------------

RUN uv sync \
    --locked \
    --no-dev \
    --no-install-project


# ============================================================
# Stage 2: Production API
# ============================================================

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# System dependency required by XGBoost
# ------------------------------------------------------------

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Create non-root user BEFORE copying application
# ------------------------------------------------------------

RUN useradd \
    --create-home \
    --shell /usr/sbin/nologin \
    appuser

# ------------------------------------------------------------
# Copy Python virtual environment
# ------------------------------------------------------------

COPY --from=builder /app/.venv /app/.venv

# ------------------------------------------------------------
# Use virtual environment
# ------------------------------------------------------------

ENV PATH="/app/.venv/bin:$PATH"

# ------------------------------------------------------------
# Copy application code
# Set ownership during COPY
# This avoids expensive chown -R layer
# ------------------------------------------------------------

COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser schemas.py .
COPY --chown=appuser:appuser src/ ./src/

# ------------------------------------------------------------
# Run as non-root
# ------------------------------------------------------------

USER appuser

# ------------------------------------------------------------
# Port
# ------------------------------------------------------------

EXPOSE 8000

# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" \
    || exit 1

# ------------------------------------------------------------
# Start FastAPI
# ------------------------------------------------------------

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]