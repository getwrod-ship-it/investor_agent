"""
app/agent/nodes/reject.py

REJECT node (terminal): static refusal for unsafe intents — no LLM call, no
retrieval, no SQL generation. The fastest and cheapest path through the
graph, and it avoids "negotiating" with the requester.
"""

import logging

from app.agent.state import AgentState

logger = logging.getLogger(__name__)

REJECT_MESSAGE = (
    "这个请求我不能处理：本系统只对业务数据库做**只读查询**，"
    "不能修改、删除或导出数据。\n\n"
    "如果你想了解数据（比如查询、统计、对比），换一个问法我马上帮你查。"
)


def reject(state: AgentState) -> AgentState:
    """
    Refuse an unsafe request with a static message (zero LLM cost).

    Args:
        state: Current AgentState — must have question set.

    Returns:
        Updated AgentState with intent="unsafe" and direct_answer set to the
        refusal message. The route layer marks this run as failed.
    """
    question = state["question"]
    logger.warning("REJECT: blocked unsafe question=%r", question[:80])

    return {
        **state,
        "intent": "unsafe",
        "direct_answer": REJECT_MESSAGE,
        "needs_clarification": False,
        "clarification_message": None,
    }
