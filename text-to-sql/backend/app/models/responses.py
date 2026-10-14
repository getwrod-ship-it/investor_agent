"""
app/models/responses.py

Pydantic response models for the FastAPI endpoints.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    session_id: str
    question: str
    sql: Optional[str]
    result: Optional[list[dict[str, Any]]]
    row_count: int
    explanation: Optional[str]
    needs_clarification: bool
    clarification_message: Optional[str]
    retry_count: int
    succeeded: bool
    trace_id: Optional[str]
    # Clarify reason — ambiguous (follow-up question) / retry_exhausted /
    # infra_failure; None when no clarification happened (incl. legacy rows).
    clarify_reason: Optional[str] = None
    # Intent guard metadata — intent is always set; the other two are only
    # populated for smalltalk / unsafe / rewritten data_query turns.
    intent: str = "data_query"
    direct_answer: Optional[str] = None
    rewritten_question: Optional[str] = None


class QueryRunSummary(BaseModel):
    """Condensed QueryRun for list responses."""

    id: str
    session_id: str
    question: str
    generated_sql: Optional[str]
    retry_count: int
    needs_clarification: bool
    succeeded: bool
    created_at: datetime


class QueryRunDetail(QueryRunSummary):
    """Full QueryRun detail including result and explanation."""

    execution_result: Optional[list[dict[str, Any]]]
    execution_error: Optional[str]
    final_explanation: Optional[str]
    clarification_message: Optional[str]
    clarify_reason: Optional[str] = None
    trace_id: Optional[str]


class SessionSummary(BaseModel):
    """One entry of the session list (frontend sidebar)."""

    session_id: str
    title: str
    turn_count: int
    last_question: Optional[str]
    created_at: datetime
    updated_at: datetime


class ChatTurnView(BaseModel):
    """
    One replayed message of a session. Field shape is aligned with
    QueryResponse so the frontend renders live replies and replayed turns
    with the same code path.

    Known trade-off: query_run never stored rewritten_question, so replayed
    turns cannot show the "系统理解为" caption (see design doc §4.5).
    """

    question: str
    sql: Optional[str]
    result: list[dict[str, Any]] = []
    row_count: int
    explanation: Optional[str]
    needs_clarification: bool
    clarification_message: Optional[str]
    clarify_reason: Optional[str] = None
    retry_count: int
    succeeded: bool
    intent: Optional[str]
    direct_answer: Optional[str]  # smalltalk turn = final_explanation
    rewritten_question: Optional[str] = None
    created_at: datetime


class SessionDetail(SessionSummary):
    """GET /sessions/{id} response — list entry plus the full replay."""

    messages: list[ChatTurnView]
