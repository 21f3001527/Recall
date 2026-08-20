# ---- Base Image ----
FROM python:3.12-slim

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Install uv ----
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# ---- Dependency layer ----
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

# ---- Copy project ----
COPY . .

# ---- Install project ----
RUN uv sync --frozen

# ---- Runtime configuration ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Render assigns PORT dynamically.
# 8501 is only the fallback for local Docker.
ENV PORT=8501

EXPOSE 8501

# ---- Healthcheck ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# ---- Start Streamlit ----
CMD ["sh", "-c", "uv run streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.fileWatcherType=none"]