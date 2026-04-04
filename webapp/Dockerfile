# syntax=docker/dockerfile:1.7

# ===== Stage 1: Builder =====
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
    --retries 20 \
    --timeout 120 \
    torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --retries 20 --timeout 120 -r requirements.txt

ARG EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    HF_HOME=/build/.model_cache \
    TRANSFORMERS_CACHE=/build/.model_cache \
    SENTENCE_TRANSFORMERS_HOME=/build/.model_cache \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN mkdir -p /build/.model_cache \
    && python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ.get('EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2'))"

# ===== Stage 2: Runtime =====
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/.model_cache /app/.model_cache

ARG EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    HF_HOME=/app/.model_cache \
    TRANSFORMERS_CACHE=/app/.model_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.model_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

COPY . .

RUN python -m playwright install --with-deps chromium

EXPOSE 8000
