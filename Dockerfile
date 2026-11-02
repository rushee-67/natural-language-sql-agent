
FROM python:3.12-slim

# Python runtime settings.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Create a non-root application user and database directory.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /data \
    && chown -R app:app /app /data

# Install runtime dependencies first for better build caching.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application and frontend source code.
COPY --chown=app:app app/ ./app/
COPY --chown=app:app frontend/ ./frontend/

# SQLite database storage location.
ENV SQLITE_DB_PATH=/data/sql_agent.db

EXPOSE 8000

# Check whether FastAPI is responding.
HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD python -c "import os, urllib.request; port=os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=3)" || exit 1

USER app

CMD ["sh", "-c", "python -m app.database.seed && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]