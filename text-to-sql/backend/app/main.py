"""
app/main.py

FastAPI application entry point.
Registers all routers, configures CORS, and runs the Qdrant embedder
(schema + knowledge base ingestion) on startup via a lifespan handler.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

# uvicorn only configures its own loggers — without this the agent nodes'
# INFO logs (GENERATE/RETRIEVE/EXECUTE/...) are swallowed at WARNING+
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup tasks (Qdrant embedding) then yield control to the app."""
    logger.info("Starting up — crawling business schema and embedding into Qdrant")
    try:
        from app.rag.embedder import embed_and_store

        embed_and_store()
        logger.info("Qdrant collections ready")
    except Exception as exc:
        # Log but do not crash startup — the app can still serve /health
        # and schema endpoints even if embedding fails (e.g. wrong connection
        # string, Qdrant unreachable, local model missing)
        logger.error("embed_and_store() failed during startup: %s", exc)

    # Warm up the long-term memory service (mem0 dual instance) so the first
    # request doesn't pay construction cost. The BGE-M3 weights are already
    # resident from embed_and_store — mem0 reuses the same singleton. A warm-up
    # failure is non-fatal: get_memory_service() retries lazily per call.
    try:
        from app.config import get_settings

        if get_settings().memory_enabled:
            from app.memory import get_memory_service

            get_memory_service()
            logger.info("Long-term memory service warmed up")
    except Exception as exc:
        logger.warning("mem0 warm-up skipped (will retry lazily): %s", exc)
    yield


app = FastAPI(
    title="Text-to-SQL Agent",
    description="Ask questions in plain language — get SQL, results, and explanations.",
    version="0.2.0",
    lifespan=lifespan,
)

# Routers
from app.api.routes import history as history_router  # noqa: E402
from app.api.routes import memory as memory_router  # noqa: E402
from app.api.routes import query as query_router  # noqa: E402
from app.api.routes import schema as schema_router  # noqa: E402
from app.api.routes import sessions as sessions_router  # noqa: E402

app.include_router(schema_router.router)
app.include_router(query_router.router)
app.include_router(history_router.router)
app.include_router(sessions_router.router)
app.include_router(memory_router.router)


@app.get("/health", tags=["ops"])
def health_check() -> dict:
    """Return service liveness status."""
    return {"status": "ok"}
