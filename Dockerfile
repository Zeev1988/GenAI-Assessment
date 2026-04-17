# syntax=docker/dockerfile:1.7

# =============================================================================
# Stage 1 - builder: create an isolated venv with the project + runtime deps.
# =============================================================================
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY common ./common
COPY form_extraction ./form_extraction
COPY medical_chatbot ./medical_chatbot

RUN pip install --upgrade pip \
 && pip install .

# =============================================================================
# Stage 2 - runtime: minimal image running the Streamlit UI as a non-root user.
# =============================================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    APP_CACHE_DIR=/app/.cache

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home /app app

COPY --from=builder /opt/venv /opt/venv
COPY common ./common
COPY form_extraction ./form_extraction
COPY medical_chatbot ./medical_chatbot

RUN mkdir -p /app/.cache && chown -R app:app /app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; \
urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read() or sys.exit(1)"

CMD ["streamlit", "run", "form_extraction/frontend/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
