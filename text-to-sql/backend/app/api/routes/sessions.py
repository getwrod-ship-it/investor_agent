"""
app/api/routes/sessions.py

Session management endpoints backing the frontend sidebar: list past
conversations, replay one in full (including unsafe refusals — replay is for
viewing), and delete with cascade (system.db rows + the checkpoint thread).
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from app.checkpoint import get_checkpointer
from app.db import get_session
from app.models.db import QueryRun
from app.models.responses import ChatTurnView, SessionDetail, SessionSummary
from app.services.history import derive_succeeded
from app.services.sessions import (
    delete_chat_session,
    get_chat_session,
    get_session_messages,
    list_chat_sessions,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])


def _run_to_turn_view(run: QueryRun) -> ChatTurnView:
    """Map a QueryRun audit row onto the replay view (QueryResponse shape)."""
    result = run.execution_result or []
    return ChatTurnView(
        question=run.question,
        sql=run.generated_sql,
        result=result,
        row_count=len(result),
        explanation=run.final_explanation,
        needs_clarification=run.needs_clarification,
        clarification_message=run.clarification_message,
        clarify_reason=run.clarify_reason,
        retry_count=run.retry_count,
        succeeded=derive_succeeded(
            run.intent,
            needs_clarification=run.needs_clarification,
            execution_error=run.execution_error,
            generated_sql=run.generated_sql,
            clarify_reason=run.clarify_reason,
        ),
        intent=run.intent,
        # smalltalk replies were mirrored into final_explanation on write;
        # unsafe turns fall back to the frontend's own refusal text
        direct_answer=run.final_explanation if run.intent == "smalltalk" else None,
        created_at=run.created_at,
    )


@router.get("", response_model=list[SessionSummary])
def get_sessions(
    limit: int = Query(default=20, ge=1, le=100, description="Max sessions to return"),
    offset: int = Query(default=0, ge=0, description="Sessions to skip"),
    session: Session = Depends(get_session),
) -> list[SessionSummary]:
    """Return the session list, most recently active first."""
    return [
        SessionSummary(
            session_id=row.id,
            title=row.title,
            turn_count=row.turn_count,
            last_question=row.last_question,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row in list_chat_sessions(session, limit=limit, offset=offset)
    ]


@router.get("/{session_id}", response_model=SessionDetail)
def get_session_detail(
    session_id: str,
    session: Session = Depends(get_session),
) -> SessionDetail:
    """Return one session's list entry plus its full chronological replay."""
    row = get_chat_session(session, session_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    messages = [
        _run_to_turn_view(run) for run in get_session_messages(session, session_id)
    ]
    return SessionDetail(
        session_id=row.id,
        title=row.title,
        turn_count=row.turn_count,
        last_question=row.last_question,
        created_at=row.created_at,
        updated_at=row.updated_at,
        messages=messages,
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    session: Session = Depends(get_session),
) -> None:
    """Delete a session: cascade its audit rows, then its checkpoint thread."""
    if not delete_chat_session(session, session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    try:
        get_checkpointer().delete_thread(session_id)
    except Exception as exc:
        # rows are already gone; a stale checkpoint thread is harmless because
        # ensure_thread_state rebuilds from query_run on the next write
        logger.warning(
            "delete_thread failed for session=%s: %s (rows already deleted)",
            session_id,
            exc,
        )
