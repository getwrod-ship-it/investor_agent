"""
app/agent/nodes/chat.py

CHAT node (terminal): answers smalltalk directly with the data-assistant
persona — one or two sentences plus a nudge back to querying. No retrieval,
no SQL generation. The reply is stored in direct_answer and mirrored into
final_explanation so it persists on the QueryRun audit record.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_chat_llm
from app.agent.prompts.chat import CHAT_SYSTEM_PROMPT, CHAT_USER_PROMPT
from app.agent.prompts.long_term import render_long_term_memory
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

_FALLBACK_ANSWER = (
    "我是 litemall 电商数据库的查询助手，可以帮你查订单、商品、用户、优惠券、售后等数据。"
    "你想查点什么？"
)


def chat(state: AgentState) -> AgentState:
    """
    Answer a smalltalk turn directly with the data-assistant persona.

    Args:
        state: Current AgentState — must have question set.

    Returns:
        Updated AgentState with direct_answer (and final_explanation) set.
        Never raises: on LLM failure a static persona answer is used.
    """
    question = state["question"]
    llm = get_chat_llm(temperature=0.7)

    # Cross-session memory is CHAT's only source of "who is this user" — the
    # rename acceptance ("我叫什么") lands here (DR-L5).
    long_term_block = render_long_term_memory(
        state.get("long_term_facts") or [], state.get("episodic_memories") or []
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=CHAT_SYSTEM_PROMPT.format(long_term_memory=long_term_block)
                ),
                HumanMessage(content=CHAT_USER_PROMPT.format(question=question)),
            ]
        )
        answer = response.content
        usage = response.usage_metadata or {}
        logger.info(
            "CHAT: tokens in=%d out=%d",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
    except Exception as exc:
        logger.error("CHAT: LLM API call failed: %s", exc)
        answer = _FALLBACK_ANSWER

    logger.info("CHAT: answered smalltalk question=%r", question[:60])

    return {
        **state,
        "direct_answer": answer,
        # Reuse the audit field so the reply persists on the QueryRun record
        # (see DESIGN_intent_guard_and_memory.md §3.3) — smalltalk counts as
        # succeeded with sql=None, computed by the route layer.
        "final_explanation": answer,
        "needs_clarification": False,
        "clarification_message": None,
    }
