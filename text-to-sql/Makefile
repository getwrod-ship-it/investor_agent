.PHONY: up down logs run frontend embed refresh tables test test-unit test-integration lint format

PY := .venv/Scripts/python.exe

# --- Qdrant container (the backend runs on the host for GPU local models) ---
up:
	docker compose up -d qdrant

down:
	docker compose down -v

logs:
	docker compose logs -f qdrant

# --- Backend (FastAPI agent) on the host ---
# Port 8888: Hyper-V/WSL reserves 7979-8778 on this machine, 8000 included
run:
	$(PY) -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8888 --reload

# --- Frontend (Streamlit) on the host ---
# Port 8889: sits next to the backend port, clear of the 7979-8778 reserve
frontend:
	$(PY) -m streamlit run frontend/app.py --server.port 8889

# --- Vector ingestion (schema crawl + knowledge base embedding) ---
embed:
	$(PY) -m app.embed_cli

refresh:
	curl -s -X POST http://localhost:8888/schema/refresh

tables:
	curl -s http://localhost:8888/schema/tables

# --- Tests ---
test:
	$(PY) -m pytest tests/ -v --tb=short

test-unit:
	$(PY) -m pytest tests/unit/ -v --tb=short

test-integration:
	$(PY) -m pytest tests/integration/ -v --tb=short

# --- Code quality ---
lint:
	$(PY) -m ruff check backend/ tests/ frontend/ mcp_server/
	$(PY) -m mypy backend/app/ --ignore-missing-imports

format:
	$(PY) -m ruff format backend/ tests/ frontend/ mcp_server/
