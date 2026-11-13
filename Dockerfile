# Graphos — Graph Intelligence Workspace
# Production image: FastAPI + DSPy engine + neon-graph web UI.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRAPHOS_PORT=8080

WORKDIR /app

# System deps kept minimal; build tools only if a wheel needs compiling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user; pre-create the data dir so a mounted volume is writable.
RUN useradd --create-home --uid 10001 graphos \
    && mkdir -p /app/graphos_data \
    && chown -R graphos:graphos /app
USER graphos

EXPOSE 8080

# Liveness probe hits the dependency-free /live route; /api/health returns 503
# when the store is down so orchestrators can pull a wedged instance.
HEALTHCHECK --interval=30s --timeout=4s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/api/health/live || exit 1

CMD ["uvicorn", "graphos.app:app", "--host", "0.0.0.0", "--port", "8080"]
