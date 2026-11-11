# CPU-only Streamlit application image (research R12). Contains no secrets, .env or model
# files: models, the HF cache and runtime data are supplied through mounted volumes.
FROM python:3.14.10-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MODEL_DIR=/app/models \
    HF_HOME=/app/.cache/huggingface \
    LABEL_SETS_FILE=/app/data/label_sets.yaml \
    INFERENCE_DEVICE=cpu

WORKDIR /app

# 1) dependencies only (cached layer); torch resolves to the +cpu wheel on Linux via uv.lock
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

# 2) the application itself
COPY src/ ./src/
COPY configs/ ./configs/
RUN uv sync --locked --no-dev

# Non-root user; directories are world-writable so `user: ${APP_UID}` overrides work with
# bind mounts on Linux hosts (research R12).
RUN useradd --uid 10001 --create-home --home-dir /home/app app \
    && mkdir -p /app/models /app/.cache/huggingface /app/data \
    && chown -R 10001:10001 /app/models /app/.cache /app/data \
    && chmod 0777 /app/models /app/.cache /app/.cache/huggingface /app/data
USER 10001

VOLUME ["/app/models", "/app/.cache/huggingface", "/app/data"]
EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["uv", "run", "--no-sync", "streamlit", "run", "src/zeroshot/ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
