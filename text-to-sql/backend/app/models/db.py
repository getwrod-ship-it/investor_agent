"""
app/models/db.py

SQLModel table definitions for the agent's internal bookkeeping tables:
  - QueryRun: one row per agent invocation (question, SQL, result, trace ID)
  - CorrectionAttempt: one row per self-correction retry
  - SchemaSnapshot: metadata about the last Qdrant schema refresh
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, JSON
from sqlmodel import Field, Relationship, SQLModel


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Return a new UUID4 as a string."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# QueryRun
# ---------------------------------------------------------------------------


class QueryRun(SQLModel, table=True):
    """One row per agent invocation — tracks the full lifecycle of a query."""

    __tablename__ = "query_run"

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    session_id: str = Field(index=True)
    question: str
    intent: Optional[str] = Field(default=None)  # data_query | smalltalk | unsafe | ambiguous
    generated_sql: Optional[str] = Field(default=None)
    execution_result: Optional[list[dict[str, Any]]] = Field(
        default=None, sa_column=Column(JSON)
    )
    execution_error: Optional[str] = Field(default=None)
    final_explanation: Optional[str] = Field(default=None)
    needs_clarification: bool = Field(default=False)
    clarification_message: Optional[str] = Field(default=None)
    clarify_reason: Optional[str] = Field(default=None)  # ambiguous | retry_exhausted | infra_failure
    retry_count: int = Field(default=0)
    trace_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)

    # One query run can have multiple correction attempts
    correction_attempts: list["CorrectionAttempt"] = Relationship(
        back_populates="query_run"
    )


# ---------------------------------------------------------------------------
# CorrectionAttempt
# ---------------------------------------------------------------------------


class CorrectionAttempt(SQLModel, table=True):
    """One row per self-correction loop iteration for a QueryRun."""

    __tablename__ = "correction_attempt"

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    query_run_id: str = Field(foreign_key="query_run.id", index=True)
    attempt_number: int
    failed_sql: str
    error_message: str
    corrected_sql: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)

    query_run: Optional[QueryRun] = Relationship(back_populates="correction_attempts")


# ---------------------------------------------------------------------------
# ChatSession
# ---------------------------------------------------------------------------


class ChatSession(SQLModel, table=True):
    """Conversation list entry — one row per chat session (lazily created)."""

    __tablename__ = "chat_session"

    id: str = Field(primary_key=True)  # = session_id (uuid, client-generated)
    title: str = Field(default="新会话")  # first question truncated; never changes
    turn_count: int = Field(default=0)  # completed turns incl. unsafe (display only)
    last_question: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow, index=True)  # list sort key


# ---------------------------------------------------------------------------
# SchemaSnapshot
# ---------------------------------------------------------------------------


class SchemaSnapshot(SQLModel, table=True):
    """Records metadata about each Qdrant schema refresh operation."""

    __tablename__ = "schema_snapshot"

    id: str = Field(default_factory=_new_uuid, primary_key=True)
    schema_collection_count: int = Field(description="Vectors in the schema collection")
    docs_collection_count: int = Field(description="Vectors in the docs collection")
    created_at: datetime = Field(default_factory=_utcnow)
