# ---- build stage: produce a wheel ---------------------------------------------------
FROM python:3.12-slim AS builder
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels .

# ---- runtime stage: slim, non-root, serving deps only -------------------------------
FROM python:3.12-slim AS runtime

ARG VERSION=dev
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="realtime-fraud-risk-engine" \
      org.opencontainers.image.description="Explainable real-time fraud risk scoring API" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FRE_ENVIRONMENT=docker \
    PORT=8000

RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels

# Optional trained artifact(s). Set FRE_MODEL_PATH=/app/models/baseline.json to use one.
COPY --chown=app:app models ./models

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=2)" || exit 1

# One worker per container: velocity state is in-process. Scale with replicas and a
# shared state store (see docs/architecture.md).
CMD ["sh", "-c", "uvicorn fraud_engine.api:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
