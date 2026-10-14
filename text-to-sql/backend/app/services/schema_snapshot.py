"""
app/services/schema_snapshot.py

Schema snapshot service.
Saves and retrieves SchemaSnapshot metadata (last crawl time, vector counts)
to/from the schema_snapshot table in the system database.
"""

import logging

from sqlmodel import Session, select

from app.models.db import SchemaSnapshot

logger = logging.getLogger(__name__)


def save_snapshot(session: Session, schema_count: int, docs_count: int) -> SchemaSnapshot:
    """
    Persist a new SchemaSnapshot row and return it.

    Args:
        session:      Active SQLModel session.
        schema_count: Number of vectors in the schema collection.
        docs_count:   Number of vectors in the docs collection.

    Returns:
        The newly created SchemaSnapshot instance.
    """
    snapshot = SchemaSnapshot(
        schema_collection_count=schema_count,
        docs_collection_count=docs_count,
    )
    session.add(snapshot)
    session.commit()
    session.refresh(snapshot)
    logger.info(
        "Saved schema snapshot: schema=%d docs=%d at %s",
        schema_count,
        docs_count,
        snapshot.created_at,
    )
    return snapshot


def get_latest_snapshot(session: Session) -> SchemaSnapshot | None:
    """
    Return the most recent SchemaSnapshot row, or None if no snapshots exist.

    Args:
        session: Active SQLModel session.
    """
    statement = select(SchemaSnapshot).order_by(SchemaSnapshot.created_at.desc()).limit(1)  # type: ignore[attr-defined]
    return session.exec(statement).first()
