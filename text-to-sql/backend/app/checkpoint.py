"""
app/checkpoint.py

Process-wide LangGraph checkpointer singleton (SqliteSaver) — the single
source of truth for conversation state. One SQLite file
(data/checkpoints.db) holds the per-thread chat memory; thread_id =
session_id, so a thread survives backend restarts.

Deliberately a separate file from the system database (data/system.db):
SqliteSaver manages its own private tables, and sharing one SQLite file
with SQLModel's create_all would cause lock contention.
"""

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_checkpointer() -> SqliteSaver:
    """
    Return the process-wide SqliteSaver instance.

    The connection is long-lived (check_same_thread=False) and shared across
    FastAPI's sync-endpoint threadpool — SqliteSaver serialises access with
    an internal threading lock. setup() creates the saver's private tables
    on first use. For multi-user deployments swap in PostgresSaver (same
    interface, different database).
    """
    settings = get_settings()
    Path(settings.checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        settings.checkpoint_db_path,
        check_same_thread=False,
    )
    saver = SqliteSaver(conn)
    saver.setup()
    logger.info("SqliteSaver ready at %s", settings.checkpoint_db_path)
    return saver
