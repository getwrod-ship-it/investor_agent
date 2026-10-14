"""
app/agent/nodes/guard.py

GUARD node (entry gate): classifies the user input into one of four intents —
data_query, smalltalk, unsafe, ambiguous — with the recent session turns as
context. Fail-open: on any LLM error the question is treated as data_query so
the main pipeline still runs.

Local decision cascade (DESIGN_local_decision_model.md DR-3): a high-confidence
answer from the local StartLux-Decision service is adopted and skips the
DeepSeek call entirely. Low confidence / service down / shadow mode falls
through to the DeepSeek JSON-mode path below, which stays byte-identical.
"""

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from app.agent.decision import decide_intent
from app.agent.llm import get_structured_llm, json_format_instructions
from app.agent.prompts.guard import GUARD_SYSTEM_PROMPT, GUARD_USER_PROMPT
from app.agent.prompts.history import render_chat_history
from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)


class IntentClassification(BaseModel):
    """Structured output from the GUARD node."""

    intent: Literal["data_query", "smalltalk", "unsafe", "ambiguous"] = Field(
        description="Classified intent of the user input"
    )
    reason: str = Field(
        description="One-sentence justification for the classification"
    )
    is_followup: bool = Field(
        default=False,
        description="True when the input is a follow-up that depends on prior turns",
    )


def guard(state: AgentState) -> AgentState:
    """
    Classify the user's question into data_query / smalltalk / unsafe / ambiguous.

    Uses the DeepSeek LLM in JSON mode to parse the response directly into an
    IntentClassification Pydantic model. Any failure falls back to data_query
    (never block a real data question because the guard broke).

    Args:
        state: Current AgentState — must have question; chat_history may be
               empty (first turn of a session).

    Returns:
        Updated AgentState with intent set.
    """
    question = state["question"]

    # --- Local decision cascade (DR-3) --------------------------------------
    # High-confidence local choice wins and skips the DeepSeek call. Anything
    # else (low confidence / service down / shadow mode) falls through to the
    # DeepSeek path below — that branch is never modified, it is the fallback.
    decision = decide_intent(
        question, state.get("chat_history") or [], state.get("history_summary")
    )
    settings = get_settings()
    if (
        decision is not None
        and not settings.decision_shadow_mode
        and decision.confidence >= settings.decision_intent_threshold
    ):
        logger.info(
            "GUARD: local decision adopted intent=%s confidence=%.3f probs=%s "
            "(DeepSeek call skipped)",
            decision.intent,
            decision.confidence,
            decision.probabilities,
        )
        return {**state, "intent": decision.intent}
    if decision is not None:
        logger.info(
            "GUARD: local decision NOT adopted (shadow=%s confidence=%.3f probs=%s) "
            "— DeepSeek path follows",
            settings.decision_shadow_mode,
            decision.confidence,
            decision.probabilities,
        )

    history_block = render_chat_history(
        state.get("chat_history") or [], summary=state.get("history_summary")
    )

    llm = get_structured_llm(IntentClassification)

    system_prompt = GUARD_SYSTEM_PROMPT.format(history=history_block)
    user_prompt = GUARD_USER_PROMPT.format(
        question=question
    ) + "\n\n" + json_format_instructions(IntentClassification)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_result: dict[str, Any] = llm.invoke(messages)
        response: IntentClassification = raw_result["parsed"]
        intent = response.intent
        reason = response.reason
        is_followup = response.is_followup
        raw_msg: AIMessage = raw_result["raw"]
        usage = raw_msg.usage_metadata or {}
        logger.info(
            "GUARD: tokens in=%d out=%d",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
    except Exception as exc:
        logger.error("GUARD: LLM API call failed: %s", exc)
        intent, reason, is_followup = "data_query", f"Guard failed, fail-open: {exc}", False

    if decision is not None and settings.decision_shadow_mode:
        logger.info(
            "GUARD[shadow]: local=%s(%.3f) vs deepseek=%s match=%s",
            decision.intent,
            decision.confidence,
            intent,
            decision.intent == intent,
        )

    logger.info(
        "GUARD: question=%r → intent=%s followup=%s reason=%s",
        question[:60],
        intent,
        is_followup,
        reason,
    )

    return {**state, "intent": intent}
