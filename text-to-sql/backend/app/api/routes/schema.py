"""
app/api/routes/schema.py

GET  /schema/tables  — list live tables/columns/FKs from the business database.
GET  /schema/status  — return collection stats and last snapshot timestamp.
POST /schema/refresh — force re-crawl and re-embed both Qdrant collections.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlmodel import Session

from app.business_db import get_business_engine
from app.config import get_settings
from app.db import get_session
from app.rag.crawler import crawl_schema
from app.rag.embedder import (
    DOCS_COLLECTION,
    SCHEMA_COLLECTION,
    _get_qdrant_client,
    embed_and_store,
)
from app.services.schema_snapshot import get_latest_snapshot, save_snapshot

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/schema", tags=["schema"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ColumnInfo(BaseModel):
    """Metadata for a single column, as reported by GET /schema/tables."""

    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool


class TableInfo(BaseModel):
    """Metadata for a single business table, as reported by GET /schema/tables."""

    table_name: str
    columns: list[ColumnInfo]
    foreign_keys: list[list[str]]
    # Each FK is [column_name, referenced_table, referenced_column]


class CollectionStats(BaseModel):
    """Stats for a single Qdrant collection."""

    name: str
    vector_count: int


class SchemaStatusResponse(BaseModel):
    """Response body for GET /schema/status."""

    collections: list[CollectionStats]
    last_snapshot_at: Optional[datetime]


class SchemaRefreshResponse(BaseModel):
    """Response body for POST /schema/refresh."""

    message: str
    schema_vectors: int
    docs_vectors: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_collection_vector_count(collection_name: str) -> int:
    """Return the current vector count for a Qdrant collection, or 0 on error."""
    try:
        client = _get_qdrant_client()
        info = client.get_collection(collection_name)
        return info.points_count or 0
    except Exception as exc:
        logger.warning("Could not get vector count for '%s': %s", collection_name, exc)
        return 0


def _crawl_live_schema():
    """Crawl the business MySQL database using the configured exclusions."""
    settings = get_settings()
    return crawl_schema(
        get_business_engine(),
        exclude_tables=settings.excluded_tables,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/tables", response_model=list[TableInfo])
def list_tables() -> list[TableInfo]:
    """
    Crawl the business database live and return its table/column/FK metadata.

    Used by the frontend schema browser and the MCP schema_tool — no cached
    copy exists; the database is always the source of truth.
    """
    schemas = _crawl_live_schema()
    return [
        TableInfo(
            table_name=s.table_name,
            columns=[
                ColumnInfo(
                    name=c.name,
                    data_type=c.data_type,
                    is_nullable=c.is_nullable,
                    is_primary_key=c.is_primary_key,
                )
                for c in s.columns
            ],
            foreign_keys=[list(fk) for fk in s.foreign_keys],
        )
        for s in schemas
    ]


@router.post("/refresh", response_model=SchemaRefreshResponse, status_code=status.HTTP_200_OK)
def refresh_schema(session: Session = Depends(get_session)) -> SchemaRefreshResponse:
    """
    Force a full re-crawl of the business database and re-embed both Qdrant
    collections.

    Deletes existing collections before re-embedding so that stale vectors
    are not left behind. Saves a SchemaSnapshot record on success.
    """
    # Drop existing collections so embed_and_store() is not skipped by idempotency check
    try:
        client = _get_qdrant_client()
        for name in (SCHEMA_COLLECTION, DOCS_COLLECTION):
            existing = {c.name for c in client.get_collections().collections}
            if name in existing:
                client.delete_collection(name)
                logger.info("Deleted collection '%s' for forced refresh", name)
    except Exception as exc:
        logger.error("Failed to delete Qdrant collections before refresh: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not reset Qdrant collections: {exc}",
        ) from exc

    # Re-embed
    try:
        embed_and_store()
    except Exception as exc:
        logger.error("embed_and_store() failed during /schema/refresh: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {exc}",
        ) from exc

    schema_count = _get_collection_vector_count(SCHEMA_COLLECTION)
    docs_count = _get_collection_vector_count(DOCS_COLLECTION)

    # Persist snapshot metadata
    save_snapshot(session, schema_count=schema_count, docs_count=docs_count)

    return SchemaRefreshResponse(
        message="Schema refresh complete",
        schema_vectors=schema_count,
        docs_vectors=docs_count,
    )


@router.get("/status", response_model=SchemaStatusResponse)
def schema_status(session: Session = Depends(get_session)) -> SchemaStatusResponse:
    """
    Return the current vector counts for both Qdrant collections and
    the timestamp of the most recent schema snapshot.
    """
    collections = [
        CollectionStats(
            name=SCHEMA_COLLECTION,
            vector_count=_get_collection_vector_count(SCHEMA_COLLECTION),
        ),
        CollectionStats(
            name=DOCS_COLLECTION,
            vector_count=_get_collection_vector_count(DOCS_COLLECTION),
        ),
    ]

    latest = get_latest_snapshot(session)
    last_snapshot_at = latest.created_at if latest else None

    return SchemaStatusResponse(collections=collections, last_snapshot_at=last_snapshot_at)
