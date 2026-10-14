"""
app/agent/nodes/generate.py

GENERATE node: prompts the LLM (DeepSeek) with retrieved schema and document
context, parses the structured response into a SQLGenerationResponse Pydantic
model (sql, reasoning, assumptions, confidence), and writes results to
AgentState.
"""

import logging
from typing import Any, Optional

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from app.agent.llm import get_structured_llm, json_format_instructions
from app.agent.prompts.generate import GENERATE_SYSTEM_PROMPT, GENERATE_USER_PROMPT
from app.agent.prompts.history import render_chat_history
from app.agent.prompts.long_term import render_long_term_memory
from app.agent.state import AgentState, ChatTurn
from app.config import get_settings

logger = logging.getLogger(__name__)

_EMPTY_HISTORY_TEXT = "（本轮是会话中的第一个问题）"


def _format_history_block(
    chat_history: list[ChatTurn], summary: Optional[str] = None
) -> str:
    """Render session memory for the GENERATE prompt (empty placeholder if none)."""
    return render_chat_history(
        chat_history, empty_text=_EMPTY_HISTORY_TEXT, summary=summary
    )


def _format_long_term_block(state: AgentState) -> str:
    """Render cross-session memory for the GENERATE prompt ('' when nothing recalled)."""
    return render_long_term_memory(
        state.get("long_term_facts") or [], state.get("episodic_memories") or []
    )


class SQLGenerationResponse(BaseModel):
    """Structured output from the GENERATE node."""

    sql: str = Field(description="The complete read-only SELECT query")
    reasoning: str = Field(description="Brief explanation of the approach taken")
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made when the question was ambiguous",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )


def generate(state: AgentState) -> AgentState:
    """
    Call the LLM to generate SQL from the retrieved context and user question.

    Uses the DeepSeek LLM in JSON mode to parse the response directly into a
    SQLGenerationResponse Pydantic model.

    Args:
        state: Current AgentState — must have question, retrieved_schema,
               and retrieved_docs populated by the RETRIEVE node.

    Returns:
        Updated AgentState with generated_sql, sql_reasoning, sql_assumptions,
        and sql_confidence set.
    """
    settings = get_settings()
    question = state["question"]
    schema_context = "\n\n".join(state["retrieved_schema"]) or "No schema context retrieved."
    docs_context = "\n\n".join(state["retrieved_docs"]) or "No business context retrieved."
    history_context = _format_history_block(
        state.get("chat_history") or [], state.get("history_summary")
    )

    llm = get_structured_llm(SQLGenerationResponse)

    system_prompt = GENERATE_SYSTEM_PROMPT.format(
        schema_context=schema_context,
        docs_context=docs_context,
        history=history_context,
        long_term_memory=_format_long_term_block(state),
        dialect=settings.business_db_dialect,
    )
    user_prompt = GENERATE_USER_PROMPT.format(question=question) + "\n\n" + json_format_instructions(
        SQLGenerationResponse
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_result: dict[str, Any] = llm.invoke(messages)
        response: SQLGenerationResponse = raw_result["parsed"]
        raw_msg: AIMessage = raw_result["raw"]
        usage = raw_msg.usage_metadata or {}
        logger.info(
            "GENERATE: tokens in=%d out=%d",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
    except Exception as exc:
        logger.error("GENERATE: LLM API call failed: %s", exc)
        return {
            **state,
            "generated_sql": None,
            "sql_reasoning": f"Generation failed: {exc}",
            "sql_assumptions": [],
            "sql_confidence": 0.0,
        }

    logger.info(
        "GENERATE: question=%r → confidence=%.2f sql_preview=%r",
        question[:60],
        response.confidence,
        response.sql[:80],
    )

    return {
        **state,
        "generated_sql": response.sql,
        "sql_reasoning": response.reasoning,
        "sql_assumptions": response.assumptions,
        "sql_confidence": response.confidence,
    }
