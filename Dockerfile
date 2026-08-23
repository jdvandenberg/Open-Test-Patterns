# Production image for Render (and any other Docker host).
# One process: FastAPI serves /api/* and the built React app.

# --- frontend ---------------------------------------------------------------
FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- backend ----------------------------------------------------------------
FROM python:3.12-slim-bookworm

# OpenImageIO's wheel still needs a few runtime libs on slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY backend ./backend
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

COPY --from=frontend /frontend/dist ./frontend/dist

ENV PYTHONPATH=/app/backend
ENV OTP_FRONTEND_DIR=/app/frontend/dist
# Render sets PORT; default 8000 for local `docker run`.
ENV PORT=8000

EXPOSE 8000

# Shell form so $PORT expands on Render.
CMD uvicorn open_test_patterns.api.app:app --host 0.0.0.0 --port ${PORT} --workers 1
