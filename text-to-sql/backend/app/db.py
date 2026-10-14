"""
app/db.py

SQLAlchemy engine and SQLModel session for the agent's PRIVATE system database
(a SQLite file). It stores only internal bookkeeping tables: query_run,
correction_attempt, schema_snapshot. The user's business database never
receives writes — see app/business_db.py for that connection.

Provides get_session() as a FastAPI dependency for injecting sessions into
route handlers.
"""

import logging
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import inspect
from sqlmodel import Session, SQLModel, create_engine

from app.config import get_settings

# Importing the models module registers the tables on SQLModel.metadata
import app.models.db  # noqa: F401

logger = logging.getLogger(__name__)


def get_engine():
    """Create and return the SQLAlchemy engine for the SQLite system database."""
    settings = get_settings()
    Path(settings.system_db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        settings.system_database_url,
        connect_args={"check_same_thread": False},
        echo=settings.app_env == "local",
    )


def _ensure_query_run_intent_column(engine) -> None:
    """
    Add the `intent` column to a pre-existing query_run table.

    SQLModel.metadata.create_all() only creates missing tables — it never
    alters existing ones. Databases created before the intent guard landed
    therefore lack the column and would fail with "no such column: intent".
    """
    inspector = inspect(engine)
    if not inspector.has_table("query_run"):
        return  # fresh database — create_all() below creates the full schema
    columns = {column["name"] for column in inspector.get_columns("query_run")}
    if "intent" not in columns:
        with engine.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE query_run ADD COLUMN intent VARCHAR")
        logger.info("Added missing 'intent' column to existing query_run table")


def _ensure_query_run_clarify_reason_column(engine) -> None:
    """
    Add the `clarify_reason` column to a pre-existing query_run table.

    Same rationale as the intent migration above: databases created before
    the clarify rework (DESIGN_clarify.md) lack the column.
    """
    inspector = inspect(engine)
    if not inspector.has_table("query_run"):
        return  # fresh database — create_all() below creates the full schema
    columns = {column["name"] for column in inspector.get_columns("query_run")}
    if "clarify_reason" not in columns:
        with engine.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE query_run ADD COLUMN clarify_reason VARCHAR")
        logger.info("Added missing 'clarify_reason' column to existing query_run table")


# Module-level engine singleton
engine = get_engine()

# Create the internal tables if they do not exist yet (idempotent)
SQLModel.metadata.create_all(engine)

# Migrate pre-existing databases that were created before the new columns
_ensure_query_run_intent_column(engine)
_ensure_query_run_clarify_reason_column(engine)


def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a SQLModel session.

    Usage:
        @app.post("/query")
        def query(session: Session = Depends(get_session)):
            ...
    """
    with Session(engine) as session:
        yield session
