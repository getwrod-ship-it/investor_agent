"""
app/agent/decision.py

Thin client for the local decision model — StartLux-Decision served at
/v1/systemone (llama-server :9081 + gguf_server :9090, sibling folder
`startlux-decision/`). See docs/DESIGN_local_decision_model.md.

Contract of every public wrapper: ANY failure (switch off, timeout, connection
refused, HTTP error, missing/malformed field) returns None, and the caller
treats None as "unknown → take the legacy path". Fail-open lives here so call
sites never repeat try/except (DR-1 / DR-2).
"""

import logging
import time
from typing import Any, NamedTuple, Optional

import httpx

from app.agent.prompts.decision import (
    INTENT_CRITERIA,
    INTENT_INSTRUCTIONS,
    LTM_NEED_INSTRUCTIONS,
    REWRITE_NEED_INSTRUCTIONS,
    WRITE_WORTHY_INSTRUCTIONS,
    build_state,
)
from app.config import get_settings

logger = logging.getLogger(__name__)

INTENT_LABELS = ("data_query", "smalltalk", "unsafe", "ambiguous")


class IntentDecision(NamedTuple):
    """Local GUARD decision. Probabilities are kept whole for the shadow-mode
    comparison log and threshold calibration (DR-7)."""

    intent: str
    confidence: float
    probabilities: dict[str, float]


def _post(
    state: dict[str, Any], questions: dict[str, dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """One POST /v1/systemone → the answers dict, or None on any failure."""
    settings = get_settings()
    if not settings.decision_enabled:
        return None
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=settings.decision_timeout_seconds) as client:
            resp = client.post(
                f"{settings.decision_base_url.rstrip('/')}/v1/systemone",
                json={"state": state, "questions": questions},
            )
        resp.raise_for_status()
        answers = resp.json().get("answers") or None
    except Exception as exc:  # noqa: BLE001 — fail-open by contract
        logger.warning("DECISION: call failed (fallback to legacy path): %s", exc)
        return None
    logger.info(
        "DECISION: qids=%s answered in %.0fms",
        sorted(questions),
        (time.perf_counter() - started) * 1000,
    )
    return answers


def _noul(state: dict[str, Any], qid: str, instructions: str) -> Optional[float]:
    """One binary question → P(true), or None on any failure."""
    answers = _post(state, {qid: {"type": "noul", "instructions": instructions}})
    if not answers:
        return None
    value = (answers.get(qid) or {}).get("noul")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        logger.warning("DECISION: malformed noul answer %r for %s — fallback", value, qid)
        return None
    return float(value)


def decide_intent(
    question: str,
    chat_history: list | None = None,
    history_summary: str | None = None,
) -> Optional[IntentDecision]:
    """① choice four-way intent classification → (intent, confidence, probs).

    The adoption threshold is NOT applied here — the guard node owns it, so
    shadow calibration sees raw confidences.
    """
    settings = get_settings()
    if not (settings.decision_enabled and settings.decision_guard_enabled):
        return None
    answers = _post(
        build_state(question, chat_history, history_summary),
        {
            "intent": {
                "type": "choice",
                "instructions": INTENT_INSTRUCTIONS,
                "criteria": INTENT_CRITERIA,
            }
        },
    )
    if not answers:
        return None
    answer = answers.get("intent") or {}
    choice = answer.get("choice")
    confidence = answer.get("confidence")
    if choice not in INTENT_LABELS or isinstance(confidence, bool) or not isinstance(
        confidence, (int, float)
    ):
        logger.warning("DECISION: malformed intent answer %r — fallback", answer)
        return None
    probabilities = {
        str(k): float(v)
        for k, v in (answer.get("probabilities") or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    return IntentDecision(str(choice), float(confidence), probabilities)


def ltm_gate_active() -> bool:
    """True only when the LTM recall gate can possibly say "skip".

    Lets the route layer skip the checkpoint read entirely when the gate is
    off — with it off the request path stays byte-identical to before.
    """
    settings = get_settings()
    return settings.decision_enabled and settings.decision_ltm_gate_enabled


def decide_ltm_needed(
    question: str, chat_history: list | None = None
) -> Optional[bool]:
    """② noul: does answering this question need long-term memory about the
    user? False (skip recall) only when P(need) < threshold — pass-biased,
    protecting the rename/convention gold standards (DR-4).
    """
    settings = get_settings()
    if not (settings.decision_enabled and settings.decision_ltm_gate_enabled):
        return None
    p = _noul(build_state(question, chat_history), "ltm_need", LTM_NEED_INSTRUCTIONS)
    if p is None:
        return None
    return p >= settings.decision_ltm_gate_threshold


def decide_write_worthy(
    question: str, answer_excerpt: str, intent: str
) -> Optional[bool]:
    """③ noul: is this turn worth persisting as a user fact? Plain 0.5 cut —
    the write gate's misjudgement cost is low (episodic leg always runs)."""
    settings = get_settings()
    if not (settings.decision_enabled and settings.decision_write_gate_enabled):
        return None
    state = build_state(question)
    state["intent"] = intent
    excerpt = str(answer_excerpt or "")[:300]
    if excerpt:
        state["answer_excerpt"] = excerpt
    p = _noul(state, "write_worthy", WRITE_WORTHY_INSTRUCTIONS)
    if p is None:
        return None
    return p >= 0.5


def decide_rewrite_needed(
    question: str, chat_history: list | None = None
) -> Optional[bool]:
    """④ noul: does this question need rewriting to stand alone? Only
    consulted on multi-turn sessions; False passes the original through and
    saves one DeepSeek call (DR-6)."""
    settings = get_settings()
    if not (settings.decision_enabled and settings.decision_rewrite_gate_enabled):
        return None
    p = _noul(build_state(question, chat_history), "rewrite_need", REWRITE_NEED_INSTRUCTIONS)
    if p is None:
        return None
    return p >= 0.5
