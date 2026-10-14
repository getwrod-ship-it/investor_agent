FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first for layer caching
COPY pyproject.toml .

# Copy application source
COPY backend/ ./backend/

# Install all dependencies (no dev extras in production)
RUN pip install --no-cache-dir -e '.'

# Copy and make the startup script executable
COPY backend/start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000

CMD ["/app/start.sh"]
