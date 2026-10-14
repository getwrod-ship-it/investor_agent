"""
app/agent/nodes/rewrite.py

REWRITE node: rebuilds the retrieval query. When the session has prior turns
the latest question may be elliptical ("那前个月呢？"); the LLM resolves
pronouns/time/subject against the K most recent turns and emits one compact
self-contained question for vector retrieval. Only RETRIEVE consumes it —
GENERATE keeps the original question plus the full history.

Local decision gate (DESIGN_local_decision_model.md DR-6): on multi-turn
sessions a self-contained question passes through without the DeepSeek call.
None (gate off / service down) → rewrite exactly as before.
"""

import logging
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from app.agent.decision import decide_rewrite_needed
from app.agent.llm import get_structured_llm, json_format_instructions
from app.agent.prompts.history import render_chat_history
from app.agent.prompts.rewrite import REWRITE_SYSTEM_PROMPT, REWRITE_USER_PROMPT
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


class QuestionRewrite(BaseModel):
    """Structured output from the REWRITE node."""

    rewritten_question: str = Field(
        description="The self-contained question to use for retrieval"
    )
    changed: bool = Field(
        default=False,
        description="True when the question was actually rewritten",
    )


def rewrite(state: AgentState) -> AgentState:
    """
    Rewrite an elliptical follow-up into a self-contained retrieval question.

    Passthrough (no LLM call) when the session has no history — the first
    question of a session pays zero extra cost. Falls back to the original
    question on any LLM error: memory must never break the main pipeline.

    Args:
        state: Current AgentState — must have question; chat_history may be
               empty (passthrough).

    Returns:
        Updated AgentState with rewritten_question set.
    """
    question = state["question"]
    chat_history = state.get("chat_history") or []

    if not chat_history:
        logger.info("REWRITE: no history — passing original question through")
        return {**state, "rewritten_question": question}

    # Decision gate (DR-6): a self-contained question in a multi-turn session
    # passes through — same return shape as the first-turn passthrough, one
    # DeepSeek call saved. None → rewrite as before.
    if decide_rewrite_needed(question, chat_history) is False:
        logger.info(
            "REWRITE: gate says self-contained — passing original question through"
        )
        return {**state, "rewritten_question": question}

    history_block = render_chat_history(
        chat_history, summary=state.get("history_summary")
    )
    llm = get_structured_llm(QuestionRewrite)

    system_prompt = REWRITE_SYSTEM_PROMPT.format(
        history=history_block,
        today=datetime.now().strftime("%Y-%m-%d"),
    )
    user_prompt = REWRITE_USER_PROMPT.format(
        question=question
    ) + "\n\n" + json_format_instructions(QuestionRewrite)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_result: dict[str, Any] = llm.invoke(messages)
        response: QuestionRewrite = raw_result["parsed"]
        rewritten = response.rewritten_question
        raw_msg: AIMessage = raw_result["raw"]
        usage = raw_msg.usage_metadata or {}
        logger.info(
            "REWRITE: tokens in=%d out=%d",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
        if not rewritten or not rewritten.strip():
            raise ValueError("empty rewritten_question returned by the LLM")
    except Exception as exc:
        logger.error(
            "REWRITE: LLM API call failed (%s) — falling back to original question", exc
        )
        return {**state, "rewritten_question": question}

    logger.info(
        "REWRITE: question=%r → rewritten=%r (changed=%s)",
        question[:60],
        rewritten[:80],
        response.changed,
    )

    return {**state, "rewritten_question": rewritten.strip()}
