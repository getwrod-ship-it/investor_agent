"""
app/agent/nodes/explain.py

EXPLAIN node (answerability gate + terminal): in a single structured LLM call,
judges whether the query result can answer the question (verdict ∈ answer /
retry / give_up, see docs/DESIGN_explain_answerability.md) and produces the
final Chinese explanation. verdict=retry writes answer_failure_reason for the
CORRECT loop instead of a final answer; the quota-exhausted final attempt
forbids retry (prompt instruction + code force-through). Any LLM failure
falls open to a static answer so the node always terminates.

With explain_answerability_enabled off (kill-switch) the legacy free-text
path below is used, byte-identical to the pre-answerability behaviour.
"""

import json
import logging
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent.llm import (
    get_chat_llm,
    get_structured_llm,
    json_format_instructions,
)
from app.agent.prompts.clarify import render_attempts
from app.agent.prompts.explain import (
    EXPLAIN_ATTEMPTS_BLOCK,
    EXPLAIN_FINAL_ATTEMPT_ADDON,
    EXPLAIN_LEGACY_SYSTEM_PROMPT,
    EXPLAIN_LEGACY_USER_PROMPT,
    EXPLAIN_SYSTEM_PROMPT,
    EXPLAIN_USER_PROMPT,
)
from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)

# Show at most this many rows in the prompt to control token usage
_MAX_PREVIEW_ROWS = 10

# Static Chinese fallback for the structured path (LLM exception / empty answer)
_EXPLAIN_FALLBACK_TEMPLATE = (
    "查询已执行成功（共 {row_count} 行结果），但生成回答失败了。请稍后重试或换个问法。"
)

# Fail-safe reason when the model returns verdict=retry without a reason
_RETRY_REASON_FALLBACK = "查询结果无法回答问题，需要换一种 SQL 查询策略再试。"


def _salvage_answer(raw_content: object, row_count: int) -> str:
    """Best-effort answer from a raw (unparsed) json_mode response.

    A loadable JSON object yields its "answer" field (covers wrapped output
    and pydantic-only validation misses). Text that still looks like JSON
    (broken quoting) is unfit for the user — fall back to the static text
    instead of dumping raw JSON at them. Plain prose is used as-is.
    """
    if not isinstance(raw_content, str) or not raw_content.strip():
        return _EXPLAIN_FALLBACK_TEMPLATE.format(row_count=row_count)
    text = raw_content.strip()
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        answer = obj.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        return _EXPLAIN_FALLBACK_TEMPLATE.format(row_count=row_count)
    if text.startswith("{"):
        return _EXPLAIN_FALLBACK_TEMPLATE.format(row_count=row_count)
    return text


class ExplainResponse(BaseModel):
    """Structured output from the EXPLAIN node (answerability gate)."""

    verdict: Literal["answer", "retry", "give_up"] = Field(
        description="answer: the result (possibly empty) answers the question; "
                    "retry: it does not, but a different SQL strategy might; "
                    "give_up: it does not and no reasonable strategy remains"
    )
    answer: Optional[str] = Field(
        default=None,
        description="Plain-language Chinese answer; required for answer/give_up, null for retry"
    )
    reason: Optional[str] = Field(
        default=None,
        description="What was returned, how it misses the question, and which SQL "
                    "strategy to try instead; required for retry, null otherwise"
    )


def explain(state: AgentState) -> AgentState:
    """
    Judge answerability and produce the final explanation (or a retry request).

    Sends the question, executed SQL, and a preview of the result rows to the
    LLM in JSON mode; the parsed verdict decides the outcome:
    - answer / give_up → final_explanation = answer (terminal);
    - retry (only before the quota is exhausted) → answer_failure_reason =
      reason, final_explanation stays None so the graph routes to CORRECT.

    Fail-open: an LLM exception or a json_mode parse failure degrades to a
    best-effort or static answer — never a dead end without a response.

    Args:
        state: Current AgentState — must have execution_result set by EXECUTE.

    Returns:
        Updated AgentState with final_explanation or answer_failure_reason set.
    """
    settings = get_settings()

    question = state["question"]
    sql = state.get("generated_sql", "")
    execution_result = state.get("execution_result") or {}
    rows = execution_result.get("rows", [])
    row_count = execution_result.get("row_count", 0)
    docs_context = "\n\n".join(state.get("retrieved_docs", [])) or "No business context."
    correction_history = state.get("correction_history", [])

    # Truncate rows for the prompt to avoid excessive token usage
    preview_rows = rows[:_MAX_PREVIEW_ROWS]
    results_preview = json.dumps(preview_rows, indent=2, default=str)
    if row_count > _MAX_PREVIEW_ROWS:
        results_preview += f"\n... ({row_count - _MAX_PREVIEW_ROWS} more rows not shown)"

    if not settings.explain_answerability_enabled:
        return _explain_legacy(
            state, question, sql, row_count, docs_context, results_preview
        )

    # Safety net (DR-4 layer 2): once the shared quota is exhausted the prompt
    # forbids retry, and the code below forces the terminal branch regardless.
    final_attempt = state.get("retry_count", 0) >= settings.max_retries

    llm = get_structured_llm(ExplainResponse)
    system_prompt = EXPLAIN_SYSTEM_PROMPT.format(docs_context=docs_context)
    if correction_history:
        # Second round on: the model needs to see what has already been tried
        system_prompt += EXPLAIN_ATTEMPTS_BLOCK.format(
            attempts=render_attempts(correction_history),
        )
    if final_attempt:
        system_prompt += EXPLAIN_FINAL_ATTEMPT_ADDON.format(
            retry_count=state.get("retry_count", 0),
        )
    user_prompt = EXPLAIN_USER_PROMPT.format(
        question=question,
        sql=sql,
        row_count=row_count,
        results_preview=results_preview,
        format_instructions=json_format_instructions(ExplainResponse),
    )

    try:
        raw_result = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        response = raw_result["parsed"]
        if response is None:
            # json_mode parse failure — best effort: salvage the answer field
            # out of the raw content before giving up on it
            explanation = _salvage_answer(raw_result["raw"].content, row_count)
            logger.warning(
                "EXPLAIN: json_mode parse failed, salvaged answer (%d chars)",
                len(explanation),
            )
            return _terminal(state, explanation)

        usage = raw_result["raw"].usage_metadata or {}
        logger.info(
            "EXPLAIN: tokens in=%d out=%d",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

        wants_retry = (not final_attempt) and response.verdict == "retry"
        if wants_retry:
            reason = response.reason or _RETRY_REASON_FALLBACK
            logger.info("EXPLAIN: verdict=retry — %s", reason[:120])
            return {
                **state,
                "final_explanation": None,
                "needs_clarification": False,
                "clarification_message": None,
                "answer_failure_reason": reason,
            }

        # answer / give_up / final-attempt force-through → terminal answer
        logger.info("EXPLAIN: verdict=%s", response.verdict)
        explanation = response.answer or _EXPLAIN_FALLBACK_TEMPLATE.format(
            row_count=row_count
        )
        return _terminal(state, explanation)
    except Exception as exc:
        logger.error("EXPLAIN: LLM API call failed: %s", exc)
        return _terminal(state, _EXPLAIN_FALLBACK_TEMPLATE.format(row_count=row_count))


def _terminal(state: AgentState, explanation: str) -> AgentState:
    """Terminal outcome: set the final answer, clear any retry request."""
    return {
        **state,
        "final_explanation": explanation,
        "needs_clarification": False,
        "clarification_message": None,
        "answer_failure_reason": None,
    }


def _explain_legacy(
    state: AgentState,
    question: str,
    sql: str,
    row_count: int,
    docs_context: str,
    results_preview: str,
) -> AgentState:
    """Kill-switch path: free-text EXPLAIN, byte-identical to the old node."""
    llm = get_chat_llm()

    system_prompt = EXPLAIN_LEGACY_SYSTEM_PROMPT.format(docs_context=docs_context)
    user_prompt = EXPLAIN_LEGACY_USER_PROMPT.format(
        question=question,
        sql=sql,
        row_count=row_count,
        results_preview=results_preview,
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        explanation = response.content
        usage = response.usage_metadata or {}
        logger.info(
            "EXPLAIN: tokens in=%d out=%d",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
        logger.info("EXPLAIN: generated explanation (%d chars)", len(explanation))
    except Exception as exc:
        logger.error("EXPLAIN: LLM API call failed: %s", exc)
        explanation = f"The query returned {row_count} row(s) but I could not generate an explanation: {exc}"

    return {
        **state,
        "final_explanation": explanation,
        "needs_clarification": False,
        "clarification_message": None,
        "answer_failure_reason": None,
    }
