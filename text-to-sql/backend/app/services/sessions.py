"""
app/services/sessions.py

ChatSession list-entry CRUD: lazy upsert after each completed turn, the
list/detail queries backing the frontend session sidebar, and a cascading
delete. The checkpoint thread itself is deleted by the sessions route via
the checkpointer (see api/routes/sessions.py).
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete, func
from sqlalchemy import select as sa_select
from sqlmodel import Session, select

from app.models.db import ChatSession, CorrectionAttempt, QueryRun

logger = logging.getLogger(__name__)


def upsert_chat_session(
    session: Session,
    *,
    session_id: str,
    question: str,
    title_max_chars: int = 20,
) -> ChatSession:
    """
    Insert or refresh the session's list entry after a completed turn.

    turn_count is recomputed from query_run instead of incremented, so the
    row self-corrects even if a previous upsert was lost to a crash. The
    title is fixed at the first question.
    """
    row = session.get(ChatSession, session_id)
    turn_count: int = session.exec(
        select(func.count())
        .select_from(QueryRun)
        .where(QueryRun.session_id == session_id)
    ).one()

    if row is None:
        row = ChatSession(id=session_id, title=question[:title_max_chars])
    row.turn_count = turn_count
    row.last_question = question
    row.updated_at = datetime.now(timezone.utc)

    session.add(row)
    session.commit()
    session.refresh(row)
    logger.info(
        "Upserted ChatSession id=%s turns=%s", session_id, row.turn_count
    )
    return row


def list_chat_sessions(
    session: Session, limit: int = 20, offset: int = 0
) -> list[ChatSession]:
    """Return sessions newest-activity first (sidebar order)."""
    statement = (
        select(ChatSession)
        .order_by(ChatSession.updated_at.desc())  # type: ignore[attr-defined]
        .offset(offset)
        .limit(limit)
    )
    return list(session.exec(statement).all())


def get_chat_session(session: Session, session_id: str) -> Optional[ChatSession]:
    """Return the ChatSession row by id (= session_id), or None."""
    return session.get(ChatSession, session_id)


def get_session_messages(session: Session, session_id: str) -> list[QueryRun]:
    """
    All turns of one session in chronological order — including unsafe turns:
    replay is for viewing, and the refusal record is part of the history
    (memory exclusion is a separate, unrelated rule).
    """
    statement = (
        select(QueryRun)
        .where(QueryRun.session_id == session_id)
        .order_by(QueryRun.created_at.asc())  # type: ignore[attr-defined]
    )
    return list(session.exec(statement).all())


def delete_chat_session(session: Session, session_id: str) -> bool:
    """
    Cascade-delete a session's rows: correction_attempt (FK child) →
    query_run → chat_session. Returns False when the session is unknown.
    """
    row = session.get(ChatSession, session_id)
    if row is None:
        return False

    run_ids = session.execute(
        sa_select(QueryRun.id).where(QueryRun.session_id == session_id)
    ).scalars().all()
    if run_ids:
        session.execute(
            delete(CorrectionAttempt).where(CorrectionAttempt.query_run_id.in_(run_ids))
        )
    session.execute(delete(QueryRun).where(QueryRun.session_id == session_id))
    session.delete(row)
    session.commit()
    logger.info("Deleted ChatSession id=%s (cascade removed %s runs)", session_id, len(run_ids))
    return True
