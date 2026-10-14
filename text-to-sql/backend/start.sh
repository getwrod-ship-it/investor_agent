#!/bin/bash
# backend/start.sh
#
# Docker entrypoint: start the API server. Internal system tables are created
# automatically on startup (SQLModel metadata.create_all) — no migrations needed.

set -e

echo "==> Starting uvicorn..."
cd /app/backend
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
