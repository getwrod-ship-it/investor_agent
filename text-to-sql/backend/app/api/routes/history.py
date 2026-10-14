"""
app/api/routes/history.py

GET /history and GET /history/{id} endpoints for retrieving past query runs.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from app.db import get_session
from app.models.responses import QueryRunDetail, QueryRunSummary
from app.services.history import derive_succeeded, get_query_run, list_query_runs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=list[QueryRunSummary])
def get_history(
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Records to skip"),
    session: Session = Depends(get_session),
) -> list[QueryRunSummary]:
    """
    Return a paginated list of past QueryRun records, newest first.

    Args:
        limit:   Maximum number of records (1–100, default 20).
        offset:  Pagination offset (default 0).
        session: Injected SQLModel session.
    """
    runs = list_query_runs(session, limit=limit, offset=offset)
    return [
        QueryRunSummary(
            id=run.id,
            session_id=run.session_id,
            question=run.question,
            generated_sql=run.generated_sql,
            retry_count=run.retry_count,
            needs_clarification=run.needs_clarification,
            succeeded=derive_succeeded(
                run.intent,
                needs_clarification=run.needs_clarification,
                execution_error=run.execution_error,
                generated_sql=run.generated_sql,
                clarify_reason=run.clarify_reason,
            ),
            created_at=run.created_at,
        )
        for run in runs
    ]


@router.get("/{run_id}", response_model=QueryRunDetail)
def get_history_by_id(
    run_id: str,
    session: Session = Depends(get_session),
) -> QueryRunDetail:
    """
    Return the full detail for a single QueryRun by its ID.

    Args:
        run_id:  UUID string of the QueryRun record.
        session: Injected SQLModel session.

    Raises:
        HTTPException 404: If no QueryRun with the given ID exists.
    """
    run = get_query_run(session, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"QueryRun '{run_id}' not found",
        )
    return QueryRunDetail(
        id=run.id,
        session_id=run.session_id,
        question=run.question,
        generated_sql=run.generated_sql,
        retry_count=run.retry_count,
        needs_clarification=run.needs_clarification,
        succeeded=derive_succeeded(
            run.intent,
            needs_clarification=run.needs_clarification,
            execution_error=run.execution_error,
            generated_sql=run.generated_sql,
            clarify_reason=run.clarify_reason,
        ),
        created_at=run.created_at,
        execution_result=run.execution_result,
        execution_error=run.execution_error,
        final_explanation=run.final_explanation,
        clarification_message=run.clarification_message,
        clarify_reason=run.clarify_reason,
        trace_id=run.trace_id,
    )
