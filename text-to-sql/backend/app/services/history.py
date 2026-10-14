"""
app/services/history.py

CRUD service functions for QueryRun records.
"""

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import func, or_
from sqlmodel import Session, select

from app.config import get_settings
from app.models.db import QueryRun


def _sanitise_value(v: Any) -> Any:
    """Convert non-JSON-serializable types returned by the database driver to JSON-safe equivalents."""
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, UUID):
        return str(v)
    return v


def _sanitise_rows(rows: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    """Make all values in execution_result rows JSON-serializable."""
    if rows is None:
        return None
    return [{k: _sanitise_value(v) for k, v in row.items()} for row in rows]

logger = logging.getLogger(__name__)


def save_query_run(
    session: Session,
    *,
    session_id: str,
    question: str,
    generated_sql: Optional[str],
    execution_result: Optional[list[dict[str, Any]]],
    execution_error: Optional[str],
    final_explanation: Optional[str],
    needs_clarification: bool,
    clarification_message: Optional[str],
    retry_count: int,
    trace_id: Optional[str],
    intent: Optional[str] = None,
    clarify_reason: Optional[str] = None,
) -> QueryRun:
    """
    Persist a new QueryRun record and return it.

    Args:
        session: Active SQLModel session.
        intent:  Guard classification ("data_query" | "smalltalk" | "unsafe" | "ambiguous").
        clarify_reason: Why CLARIFY fired ("ambiguous" | "retry_exhausted" |
                       "infra_failure"), None when it did not.
        All other args map 1-to-1 to QueryRun fields.

    Returns:
        The newly created and refreshed QueryRun instance.
    """
    run = QueryRun(
        session_id=session_id,
        question=question,
        intent=intent,
        generated_sql=generated_sql,
        execution_result=_sanitise_rows(execution_result),
        execution_error=execution_error,
        final_explanation=final_explanation,
        needs_clarification=needs_clarification,
        clarification_message=clarification_message,
        clarify_reason=clarify_reason,
        retry_count=retry_count,
        trace_id=trace_id,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    logger.info(
        "Saved QueryRun id=%s session_id=%s intent=%s", run.id, session_id, intent
    )
    return run


def get_query_run(session: Session, run_id: str) -> Optional[QueryRun]:
    """
    Return a QueryRun by primary key, or None if not found.

    Args:
        session: Active SQLModel session.
        run_id:  Primary key UUID string.
    """
    return session.get(QueryRun, run_id)


def list_query_runs(session: Session, limit: int = 20, offset: int = 0) -> list[QueryRun]:
    """
    Return a paginated list of QueryRun records, newest first.

    Args:
        session: Active SQLModel session.
        limit:   Maximum number of records to return (default 20).
        offset:  Number of records to skip for pagination (default 0).
    """
    statement = (
        select(QueryRun)
        .order_by(QueryRun.created_at.desc())  # type: ignore[attr-defined]
        .offset(offset)
        .limit(limit)
    )
    return list(session.exec(statement).all())


def get_recent_session_turns(
    session: Session, session_id: str, k: Optional[int] = None
) -> list[QueryRun]:
    """
    Return the most recent k turns of one session, in chronological order.

    Unsafe turns are excluded — their raw text must never be fed back into
    later prompts (prompt-pollution defence). Rows with intent IS NULL are
    legacy records written before the guard existed and are kept.

    Args:
        session:    Active SQLModel session.
        session_id: Session whose turns should be loaded.
        k:          Window size; defaults to settings.memory_window_size.
    """
    if k is None:
        k = get_settings().memory_window_size
    statement = (
        select(QueryRun)
        .where(QueryRun.session_id == session_id)
        .where(or_(QueryRun.intent != "unsafe", QueryRun.intent.is_(None)))  # type: ignore[union-attr]
        .order_by(QueryRun.created_at.desc())  # type: ignore[attr-defined]
        .limit(k)
    )
    turns = list(session.exec(statement).all())
    turns.reverse()  # newest-last, so prompts read chronologically
    return turns


def derive_succeeded(
    intent: Optional[str],
    *,
    needs_clarification: bool,
    execution_error: Optional[str],
    generated_sql: Optional[str],
    clarify_reason: Optional[str] = None,
) -> bool:
    """
    Intent-aware success semantics, shared by POST /query and the session
    replay views: smalltalk is a successful answer without SQL; unsafe is a
    refusal (failure); data_query needs SQL, no clarification and no error.
    An ambiguous follow-up question is a normal interaction turn, not a
    failure — the user was asked to rephrase, which is the intended outcome.
    """
    if intent == "smalltalk":
        return True
    if intent == "unsafe":
        return False
    if needs_clarification:
        return clarify_reason == "ambiguous"
    return execution_error is None and generated_sql is not None


def count_memory_turns(session: Session, session_id: str) -> int:
    """
    Count a session's memory-eligible turns (non-unsafe) in query_run.

    The self-heal check in services/conversation.py compares this against the
    checkpoint's memory_turn_count watermark. Rows with intent IS NULL are
    legacy records written before the guard existed and are kept.
    """
    statement = (
        select(func.count())
        .select_from(QueryRun)
        .where(QueryRun.session_id == session_id)
        .where(or_(QueryRun.intent != "unsafe", QueryRun.intent.is_(None)))  # type: ignore[union-attr]
    )
    return session.exec(statement).one()


def get_turns_before(
    session: Session, session_id: str, window: int
) -> list[QueryRun]:
    """
    Return the session's non-unsafe turns OLDER than the most recent `window`
    turns, in chronological order — the range a rebuilt rolling summary must
    cover when the memory window is reconstructed from query_run.
    """
    statement = (
        select(QueryRun)
        .where(QueryRun.session_id == session_id)
        .where(or_(QueryRun.intent != "unsafe", QueryRun.intent.is_(None)))  # type: ignore[union-attr]
        .order_by(QueryRun.created_at.desc())  # type: ignore[attr-defined]
        .offset(window)
    )
    turns = list(session.exec(statement).all())
    turns.reverse()  # chronological, oldest first
    return turns
