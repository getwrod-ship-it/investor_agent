"""
app/services/conversation.py

Turn-end memory maintenance for LangGraph threads (checkpoint-backed).
The route layer owns the memory policy; graph nodes stay read-only
consumers (see DESIGN_session_checkpoint_and_memory.md §4.3/§4.4).

  - ensure_thread_state: self-heal before every invoke — when the checkpoint
    watermark is stale (legacy sessions predating checkpoints, a crash
    inside the double-write window, or a lost checkpoints.db), rebuild the
    memory window from query_run, the audit source of truth.
  - append_turn: after every invoke — append the completed turn to the
    memory window, trim to k, and bump the watermark.
"""

import logging
from typing import Any, Optional, Protocol

from sqlmodel import Session

from app.agent.llm import get_chat_llm
from app.agent.prompts.history import render_chat_history
from app.agent.prompts.summary import SUMMARY_SYSTEM_PROMPT
from app.agent.state import AgentState, ChatTurn
from app.config import get_settings
from app.services.history import (
    count_memory_turns,
    get_recent_session_turns,
    get_turns_before,
)

logger = logging.getLogger(__name__)


class GraphProtocol(Protocol):
    """The compiled-graph surface this module relies on."""

    def invoke(self, state: AgentState, config: Optional[dict] = None) -> dict: ...

    def get_state(self, config: dict) -> Any: ...

    def update_state(self, config: dict, values: dict) -> Any: ...


def _run_to_chat_turn(run: Any) -> ChatTurn:
    """Map a QueryRun row onto the compact ChatTurn memory shape."""
    return {
        "question": run.question,
        "sql": run.generated_sql,                  # smalltalk turns: None
        # clarify turns carry their message in clarification_message — without
        # it the follow-up answer would arrive with no memory of what was asked
        "answer": (run.final_explanation or run.clarification_message or "")[:200],
    }


def ensure_thread_state(
    db_session: Session, graph: GraphProtocol, config: dict, session_id: str
) -> None:
    """
    Reconcile checkpoint memory with the query_run audit log before an invoke.

    Invariant: memory_turn_count == count of the session's non-unsafe
    query_run rows. On mismatch the window is rebuilt from query_run so the
    three real drift scenarios (legacy data, crash between the two writes,
    lost checkpoints.db) all converge to a consistent state. The summary is
    best-effort: it is reused when still valid and left None otherwise.
    """
    n = count_memory_turns(db_session, session_id)
    if n == 0:
        return  # brand-new (or unsafe-only) session — nothing to reconcile

    snap = graph.get_state(config)
    memory_turn_count = snap.values.get("memory_turn_count", 0)
    if memory_turn_count == n:
        return

    k = get_settings().memory_window_size
    runs = get_recent_session_turns(db_session, session_id, k=min(k, n))
    history = [_run_to_chat_turn(run) for run in runs]
    summary = snap.values.get("history_summary")
    older = get_turns_before(db_session, session_id, window=len(history))
    if older and (summary is None or memory_turn_count == 0):
        summary = summarize_once(older)  # fail-open: None on error

    graph.update_state(
        config,
        {
            "chat_history": history,
            "history_summary": summary,
            "memory_turn_count": n,
        },
    )
    logger.info(
        "SELF-HEAL: rebuilt thread memory for session=%s (watermark %s → %s turns)",
        session_id,
        memory_turn_count,
        n,
    )


def append_turn(
    graph: GraphProtocol, config: dict, final_state: AgentState
) -> None:
    """
    Append the completed turn to thread memory and enforce the window.

    Unsafe turns are never appended (write-time filtering, DR-6) — their raw
    text must not leak into later prompts. Smalltalk turns append with
    sql=None, exactly like the old read-time loader produced.
    """
    if (final_state.get("intent") or "data_query") == "unsafe":
        return

    turn: ChatTurn = {
        "question": final_state["question"],
        "sql": final_state.get("generated_sql"),          # smalltalk turns: None
        # clarify turns carry their message in clarification_message — without
        # it the follow-up answer would arrive with no memory of what was asked
        "answer": (
            final_state.get("final_explanation")
            or final_state.get("clarification_message")
            or ""
        )[:200],
    }

    snap = graph.get_state(config)
    history: list[ChatTurn] = list(snap.values.get("chat_history") or []) + [turn]
    summary = snap.values.get("history_summary")
    k = get_settings().memory_window_size

    if len(history) > k:
        fallen = history[:-k]  # usually exactly one turn
        history = history[-k:]
        if get_settings().memory_summary_enabled:
            summary = update_summary(summary, fallen)

    graph.update_state(
        config,
        {
            "chat_history": history,
            "history_summary": summary,
            "memory_turn_count": snap.values.get("memory_turn_count", 0) + 1,
        },
    )


def _compress(old_summary: Optional[str], turns: list[ChatTurn]) -> str:
    """Fold turns (plus the old summary) into one digest via DeepSeek.

    Raises on any LLM failure — callers apply their own fail-open policy.
    """
    settings = get_settings()
    llm = get_chat_llm(temperature=0)
    prompt = SUMMARY_SYSTEM_PROMPT.format(
        old_summary=old_summary or "（无）",
        fallen_turns=render_chat_history(turns, empty_text="（无）"),
        max_chars=settings.memory_summary_max_chars,
    )
    response = llm.invoke(prompt)
    text = (response.content or "").strip()
    if not text:
        raise ValueError("empty summary returned by the LLM")
    return text[: settings.memory_summary_max_chars]


def update_summary(
    old_summary: Optional[str], fallen_turns: list[ChatTurn]
) -> Optional[str]:
    """
    Fold the turns that just fell out of the window into the rolling summary.

    Fail-open: on any LLM error the old summary is returned unchanged —
    compression must never block the conversation. The fallen turns lose
    detail this round; the next overflow retries.
    """
    try:
        return _compress(old_summary, fallen_turns)
    except Exception as exc:
        logger.warning(
            "SUMMARY: rolling fold failed (%s) — keeping previous summary", exc
        )
        return old_summary


def summarize_once(runs: list[Any]) -> Optional[str]:
    """
    One-shot compression of pre-window QueryRun turns (self-heal rebuild).

    Returns None on failure — memory degrades to the bare window, never
    blocks the conversation.
    """
    try:
        return _compress(None, [_run_to_chat_turn(run) for run in runs])
    except Exception as exc:
        logger.warning(
            "SUMMARY: one-shot rebuild failed (%s) — continuing without summary", exc
        )
        return None
