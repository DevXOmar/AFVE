# ============================================================
# AFVE — Production Dockerfile
# Multi-stage build: builder → runtime
# Target: linux/amd64  (add --platform=linux/arm64 for M-series)
# ============================================================

# ── Stage 1: dependency builder ─────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first (layer cache optimisation)
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: lean runtime image ─────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="platform-ai@company.com"
LABEL org.opencontainers.image.title="Agentic Factual Verification Engine"
LABEL org.opencontainers.image.version="1.0.0"

# Non-root user for security hardening
RUN groupadd --gid 1001 afve \
    && useradd --uid 1001 --gid afve --shell /bin/bash --create-home afve

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=afve:afve . .

# HuggingFace cache — pre-warm during build if models are available
# (Remove or adjust if you mount models from a volume at runtime)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
RUN mkdir -p /app/.cache/huggingface && chown -R afve:afve /app/.cache

# Switch to non-root
USER afve

# ── Runtime environment defaults ────────────────────────────
# All AFVE_ prefixed vars are read by config.py (pydantic-settings)
ENV AFVE_LOG_LEVEL=INFO \
    AFVE_DEBUG=false \
    AFVE_METRICS_ENABLED=true \
    AFVE_REDIS_URL=redis://redis:6379/0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# FastAPI port
EXPOSE 8000

# Prometheus metrics scrape port
EXPOSE 9090

# ── Health check ────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ──────────────────────────────────────────────
# Uvicorn with multiple workers — for true multi-process scaling,
# use Gunicorn with the UvicornWorker class (see SCALABILITY notes).
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--access-log", \
     "--log-level", "info"]
