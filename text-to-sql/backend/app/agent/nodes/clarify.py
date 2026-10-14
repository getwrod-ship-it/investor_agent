"""
app/agent/nodes/clarify.py

CLARIFY node (terminal), three paths (DESIGN_clarify.md):
  1. ambiguous       — the guard flagged the question as unclear; the LLM
                       generates one targeted follow-up question.
  2. infra_failure   — retries exhausted but no SQL was ever generated: the
                       model service itself is likely down. Detected by pure
                       rules, answered by a static message, NO LLM call (it
                       would fail too).
  3. retry_exhausted — the LLM reads the real correction history and explains
                       the failure cause plus how to rephrase.

Every LLM path fails open to a static Chinese template, so CLARIFY can never
raise and always produces a message for the user.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_chat_llm
from app.agent.prompts.clarify import (
    CLARIFY_AMBIGUOUS_SYSTEM_PROMPT,
    CLARIFY_AMBIGUOUS_USER_PROMPT,
    CLARIFY_DIAGNOSIS_SYSTEM_PROMPT,
    CLARIFY_DIAGNOSIS_USER_PROMPT,
    render_ambiguous_history,
    render_attempts,
    render_schema_context,
)
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

_NO_SQL_ERROR = "No SQL was generated"

INFRA_FAILURE_MESSAGE = (
    "模型服务暂时不可用，这次没能生成查询。请稍后重试；"
    "如果持续出现，请联系管理员查看服务状态。"
)

AMBIGUOUS_FALLBACK = (
    "为了准确查询，请补充：① 查询对象（哪个店铺 / 商品 / 用户群体）；"
    "② 统计口径（如按销售额还是按销量）；③ 需要的时间范围。"
)

RETRY_FALLBACK_TEMPLATE = (
    "尝试了 {retry_count} 次都未能生成可执行的查询。最后错误：{last_error}\n\n"
    "你可以尝试：\n"
    "• 换更明确的表名 / 列名（可在表结构浏览器中确认）\n"
    "• 把复杂问题拆成几步分别提问\n"
    "• 显式加上时间、品类、地区等筛选条件"
)


def _looks_like_infra_failure(state: AgentState) -> bool:
    """
    True when SQL was never generated across all attempts.

    Every correction record having an empty failed_sql with the
    "No SQL was generated" error means generation itself failed every time —
    the failure is in the model service, not in SQL logic. The generation
    exception text cannot be used here: CORRECT overwrites sql_reasoning, so
    by the time CLARIFY runs only this structural signal survives.
    """
    history = state.get("correction_history") or []
    if not history:
        return False
    return all(not (record.get("failed_sql") or "").strip() for record in history) and all(
        record.get("error_message") == _NO_SQL_ERROR for record in history
    )


def _generate_followup_question(state: AgentState) -> str:
    """LLM path for ambiguous: one targeted follow-up question, static fallback."""
    try:
        llm = get_chat_llm(temperature=0)
        response = llm.invoke(
            [
                SystemMessage(
                    content=CLARIFY_AMBIGUOUS_SYSTEM_PROMPT.format(
                        history=render_ambiguous_history(state.get("chat_history") or [])
                    )
                ),
                HumanMessage(
                    content=CLARIFY_AMBIGUOUS_USER_PROMPT.format(
                        question=state["question"]
                    )
                ),
            ]
        )
        text = (response.content or "").strip()
        return text or AMBIGUOUS_FALLBACK
    except Exception as exc:
        logger.warning("CLARIFY(ambiguous): LLM failed (%s) — using fallback", exc)
        return AMBIGUOUS_FALLBACK


def _diagnose_failure(state: AgentState) -> str:
    """LLM path for retry_exhausted: failure diagnosis, Chinese fallback template."""
    retry_count = state.get("retry_count", 0)
    last_error = (
        state.get("execution_error")
        or state.get("validation_error")
        or "unknown error"
    )
    try:
        llm = get_chat_llm(temperature=0)
        response = llm.invoke(
            [
                SystemMessage(content=CLARIFY_DIAGNOSIS_SYSTEM_PROMPT),
                HumanMessage(
                    content=CLARIFY_DIAGNOSIS_USER_PROMPT.format(
                        question=state["question"],
                        attempts=render_attempts(state.get("correction_history") or []),
                        last_error=last_error,
                        schema_context=render_schema_context(
                            state.get("retrieved_schema") or []
                        ),
                    )
                ),
            ]
        )
        text = (response.content or "").strip()
        if text:
            return text
        raise ValueError("empty diagnosis returned by the LLM")
    except Exception as exc:
        logger.warning("CLARIFY(retry): LLM failed (%s) — using fallback", exc)
        return RETRY_FALLBACK_TEMPLATE.format(
            retry_count=retry_count, last_error=last_error
        )


def clarify(state: AgentState) -> AgentState:
    """
    Terminal node: tell the user why the pipeline stopped and what to do next.

    Dispatches by state:
      intent == "ambiguous"           → follow-up question (LLM, fallback static)
      infra failure (rule-detected)   → static service-unavailable message
      otherwise (retries exhausted)   → LLM failure diagnosis (fallback template)

    Returns:
        Updated AgentState with needs_clarification=True, clarification_message
        set, and clarify_reason identifying which path fired.
    """
    if state.get("intent") == "ambiguous":
        reason, message = "ambiguous", _generate_followup_question(state)
    elif _looks_like_infra_failure(state):
        reason, message = "infra_failure", INFRA_FAILURE_MESSAGE
    else:
        reason, message = "retry_exhausted", _diagnose_failure(state)

    logger.warning(
        "CLARIFY: question=%r reason=%s message_preview=%r",
        state["question"][:60],
        reason,
        message[:100],
    )

    return {
        **state,
        "needs_clarification": True,
        "clarification_message": message,
        "clarify_reason": reason,
        "final_explanation": None,
    }
