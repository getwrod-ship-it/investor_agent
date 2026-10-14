"""
app/api/routes/query.py

POST /query endpoint: accepts a QueryRequest, invokes the LangGraph agent,
persists the QueryRun record, and returns a QueryResponse.
"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlmodel import Session

from app.agent.decision import (
    decide_ltm_needed,
    decide_write_worthy,
    ltm_gate_active,
)
from app.agent.graph import graph
from app.agent.state import AgentState
from app.config import get_settings
from app.db import get_session
from app.memory import RecallResult, get_memory_service
from app.models.requests import QueryRequest
from app.models.responses import QueryResponse
from app.services.conversation import append_turn, ensure_thread_state
from app.services.history import derive_succeeded, save_query_run
from app.services.sessions import upsert_chat_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


def _write_turn_with_gate(
    memory_service: Any,
    *,
    question: str,
    intent: str,
    sql: str | None,
    answer: str | None,
    session_id: str,
    trace_id: str | None,
) -> None:
    """
    Background wrapper around memory_service.write_turn (DR-5).

    The write-worthiness gate runs INSIDE the background task: its ~100ms
    decision never lands on the request path. worthy is None (gate off /
    service down / unknown) → full write, exactly the legacy behaviour.
    """
    settings = get_settings()
    if settings.decision_enabled and settings.decision_write_gate_enabled:
        excerpt = (answer or "")[:300]
        worthy = decide_write_worthy(question, excerpt, intent)
    else:
        worthy = None
    memory_service.write_turn(
        question=question,
        intent=intent,
        sql=sql,
        answer=answer,
        session_id=session_id,
        trace_id=trace_id,
        skip_facts=worthy is False,
    )


@router.post("", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def run_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> QueryResponse:
    """
    Invoke the LangGraph agent with the user's question.

    Builds the initial AgentState, runs the compiled graph, persists the
    result to the system database, and returns a QueryResponse. On an
    unexpected server error the endpoint returns HTTP 500.

    Args:
        request: QueryRequest with question and optional session_id.
        background_tasks: FastAPI background tasks — the long-term memory
            write runs here after the response is sent (async by design,
            DR-L6: mem0.add costs 1~2 LLM calls).
        session: Injected SQLModel database session.

    Returns:
        QueryResponse with SQL, results, explanation, and metadata.
    """
    trace_id = str(uuid.uuid4())

    # thread_id = session_id: the checkpointer carries chat memory across
    # requests AND process restarts. ensure_thread_state self-heals the
    # memory first (legacy sessions / crash inside the double-write window).
    config = {"configurable": {"thread_id": request.session_id}}
    ensure_thread_state(session, graph, config, request.session_id)

    # Long-term recall (parallel facts+episodes, fail-open, ms-level local
    # vector search — never blocks the request even if mem0/Qdrant is down).
    memory_service = get_memory_service()
    # LTM recall gate (DESIGN_local_decision_model.md DR-4): when the local
    # decision service says this turn clearly needs no cross-session memory,
    # skip both legs. None (= gate off / service down / low confidence) →
    # recall exactly as before. ltm_gate_active() keeps the checkpoint read
    # off the request path entirely when the gate is disabled.
    if ltm_gate_active():
        snapshot = graph.get_state(config)
        chat_window = snapshot.values.get("chat_history") or []
        if decide_ltm_needed(request.question, chat_window) is False:
            logger.info("LTM-GATE: skip recall for question=%r", request.question[:60])
            memories = RecallResult(facts=[], episodes=[])
        else:
            memories = memory_service.recall(request.question)
    else:
        memories = memory_service.recall(request.question)

    initial_state: AgentState = {
        "question": request.question,
        "session_id": request.session_id,
        "intent": "data_query",
        "direct_answer": None,
        "rewritten_question": None,
        "retrieved_schema": [],
        "retrieved_docs": [],
        "generated_sql": None,
        "sql_reasoning": None,
        "sql_assumptions": [],
        "sql_confidence": 0.0,
        "validation_error": None,
        "execution_result": None,
        "execution_error": None,
        "retry_count": 0,
        "correction_history": [],
        "needs_clarification": False,
        "clarification_message": None,
        "clarify_reason": None,
        "final_explanation": None,
        "answer_failure_reason": None,
        "trace_id": trace_id,
        "long_term_facts": memories.facts,
        "episodic_memories": memories.episodes,
        # chat_history / history_summary / memory_turn_count are deliberately
        # absent: with a checkpointer the input only updates the channels it
        # provides — memory persists from the previous checkpoint of this
        # thread. Every per-turn field above must stay explicitly reset or
        # the previous turn's residue leaks into this one.
    }

    try:
        final_state: AgentState = graph.invoke(initial_state, config)
    except Exception as exc:
        logger.exception("Graph invocation failed for question=%r: %s", request.question, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {exc}",
        ) from exc

    intent = final_state.get("intent") or "data_query"

    # Memory maintenance happens before the audit write: a crash between the
    # two is repaired by ensure_thread_state on the next request.
    append_turn(graph, config, final_state)

    execution_result = final_state.get("execution_result") or {}
    rows = execution_result.get("rows", [])
    row_count = execution_result.get("row_count", 0)

    # succeeded semantics per intent: smalltalk is a successful answer without
    # SQL; unsafe is a refusal (failure); an ambiguous follow-up question is a
    # normal interaction turn; data_query keeps the original logic.
    succeeded = derive_succeeded(
        intent,
        needs_clarification=final_state.get("needs_clarification", False),
        execution_error=final_state.get("execution_error"),
        generated_sql=final_state.get("generated_sql"),
        clarify_reason=final_state.get("clarify_reason"),
    )

    # Persist to the system database (never to the business database)
    save_query_run(
        session,
        session_id=final_state["session_id"],
        question=final_state["question"],
        generated_sql=final_state.get("generated_sql"),
        execution_result=rows,
        execution_error=final_state.get("execution_error"),
        final_explanation=final_state.get("final_explanation"),
        needs_clarification=final_state.get("needs_clarification", False),
        clarification_message=final_state.get("clarification_message"),
        retry_count=final_state.get("retry_count", 0),
        trace_id=trace_id,
        intent=intent,
        clarify_reason=final_state.get("clarify_reason"),
    )

    # Refresh the session list entry (lazily created on the first question)
    upsert_chat_session(
        session,
        session_id=final_state["session_id"],
        question=final_state["question"],
    )

    # Long-term memory write, AFTER the response payload is built: runs as a
    # background task (1~2 LLM calls for facts extraction). Unsafe turns are
    # not memorized — same write-time filtering as the short-term memory
    # (DR-6 / DR-L6). The write gate (DR-5) decides inside the background
    # task whether the facts leg is worth its LLM calls. write_turn never
    # raises.
    if intent != "unsafe":
        background_tasks.add_task(
            _write_turn_with_gate,
            memory_service,
            question=request.question,
            intent=intent,
            sql=final_state.get("generated_sql"),
            answer=final_state.get("final_explanation")
            or final_state.get("direct_answer"),
            session_id=request.session_id,
            trace_id=trace_id,
        )

    return QueryResponse(
        session_id=final_state["session_id"],
        question=final_state["question"],
        sql=final_state.get("generated_sql"),
        result=rows,
        row_count=row_count,
        explanation=final_state.get("final_explanation"),
        needs_clarification=final_state.get("needs_clarification", False),
        clarification_message=final_state.get("clarification_message"),
        retry_count=final_state.get("retry_count", 0),
        succeeded=succeeded,
        trace_id=trace_id,
        intent=intent,
        clarify_reason=final_state.get("clarify_reason"),
        direct_answer=final_state.get("direct_answer"),
        rewritten_question=final_state.get("rewritten_question"),
    )
