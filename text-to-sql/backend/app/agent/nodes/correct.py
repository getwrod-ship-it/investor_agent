"""
app/agent/nodes/correct.py

CORRECT node: increments retry_count, appends the failed SQL and error to
correction_history, then re-prompts the LLM (DeepSeek) with the full error
context to generate a corrected SQL statement.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.llm import get_structured_llm, json_format_instructions
from app.agent.nodes.generate import SQLGenerationResponse
from app.agent.prompts.clarify import render_attempts
from app.agent.prompts.correct import (
    CORRECT_ATTEMPTS_BLOCK,
    CORRECT_SYSTEM_PROMPT,
    CORRECT_USER_PROMPT,
)
from app.agent.state import AgentState, CorrectionRecord
from app.config import get_settings

logger = logging.getLogger(__name__)


def correct(state: AgentState) -> AgentState:
    """
    Self-correction node: diagnoses the last failure and asks the LLM for a fix.

    Records the failed attempt in correction_history, increments retry_count,
    then calls the LLM with the original question, schema/doc context, failed
    SQL, and error message. The corrected SQL is written back to generated_sql
    so the graph can route through VALIDATE → EXECUTE again.

    The error message comes from the first set channel: validation_error,
    execution_error, or answer_failure_reason (semantic failure — the result
    executed fine but does not answer the question; see
    docs/DESIGN_explain_answerability.md). Earlier attempts are injected into
    the prompt so the model changes strategy instead of repeating itself.

    Args:
        state: Current AgentState — must have generated_sql and one of
               validation_error / execution_error / answer_failure_reason set.

    Returns:
        Updated AgentState with incremented retry_count, updated
        correction_history, and new generated_sql (or None on LLM failure).
    """
    settings = get_settings()

    failed_sql = state.get("generated_sql") or ""
    error_message = (
        state.get("validation_error")
        or state.get("execution_error")
        or state.get("answer_failure_reason")
        or "Unknown error"
    )
    retry_count = state.get("retry_count", 0) + 1
    attempt_number = retry_count

    # Record this attempt before trying to fix it
    record: CorrectionRecord = {
        "attempt_number": attempt_number,
        "failed_sql": failed_sql,
        "error_message": error_message,
        "corrected_sql": None,
    }
    correction_history = list(state.get("correction_history", [])) + [record]

    logger.info(
        "CORRECT: attempt #%d — error=%r sql_preview=%r",
        attempt_number,
        error_message[:100],
        failed_sql[:80],
    )

    schema_context = "\n\n".join(state.get("retrieved_schema", [])) or "No schema context."
    docs_context = "\n\n".join(state.get("retrieved_docs", [])) or "No business context."

    llm = get_structured_llm(SQLGenerationResponse)

    system_prompt = CORRECT_SYSTEM_PROMPT.format(
        schema_context=schema_context,
        docs_context=docs_context,
        dialect=settings.business_db_dialect,
    )
    user_prompt = CORRECT_USER_PROMPT.format(
        question=state["question"],
        attempt_number=attempt_number,
        failed_sql=failed_sql,
        error_message=error_message,
    )
    # Earlier attempts (everything but the one being fixed above) so the
    # model avoids repeated strategies on semantic retries
    if len(correction_history) > 1:
        user_prompt += CORRECT_ATTEMPTS_BLOCK.format(
            attempts=render_attempts(correction_history[:-1])
        )
    user_prompt += "\n\n" + json_format_instructions(SQLGenerationResponse)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = None
    try:
        raw_result: dict[str, Any] = llm.invoke(messages)
        response = raw_result["parsed"]
        raw_msg: AIMessage = raw_result["raw"]
        usage = raw_msg.usage_metadata or {}
        corrected_sql = response.sql
        correction_history[-1]["corrected_sql"] = corrected_sql
        logger.info(
            "CORRECT: tokens in=%d out=%d",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
        logger.info("CORRECT: corrected sql_preview=%r confidence=%.2f", corrected_sql[:80], response.confidence)
    except Exception as exc:
        logger.error("CORRECT: LLM API call failed: %s", exc)
        corrected_sql = None

    return {
        **state,
        "generated_sql": corrected_sql,
        "sql_reasoning": response.reasoning if response else None,
        "sql_assumptions": response.assumptions if response else [],
        "sql_confidence": response.confidence if response else 0.0,
        "validation_error": None,
        "execution_error": None,
        "answer_failure_reason": None,
        "retry_count": retry_count,
        "correction_history": correction_history,
    }
