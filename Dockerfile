# syntax=docker/dockerfile:1

############################
# Builder: build wheels (offline-friendly)
############################
FROM python:3.9-slim-bullseye AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/api.lock.txt requirements/api.lock.txt

RUN python -m pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements/api.lock.txt

############################
# Runtime: minimal image, non-root, offline install
############################
FROM python:3.9-slim-bullseye AS runtime

# --- TLS/CA certificates for outbound HTTPS/TLS (MongoDB Atlas needs this) ---
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && update-ca-certificates \
  && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN useradd -m -u 10001 appuser

COPY --from=builder /wheels /wheels
COPY requirements/api.lock.txt requirements/api.lock.txt

RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir --no-index --find-links=/wheels --require-hashes -r requirements/api.lock.txt \
 && rm -rf /wheels

COPY pyproject.toml ./pyproject.toml
COPY src ./src
COPY --chown=appuser:appuser apps ./apps

# Curated runtime artifacts (allowed by .dockerignore)
COPY --chown=appuser:appuser data ./data

RUN python -m pip install --no-cache-dir --no-deps .

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health').read()" || exit 1

CMD ["sh", "-c", "python -m apps.api.app.startup_diagnostics || true; uvicorn apps.api.app.main:app --host 0.0.0.0 --port ${PORT}"]
