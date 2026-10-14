"""
app/agent/graph.py

Assembles the full 11-node LangGraph StateGraph: GUARD → (CHAT | REJECT |
CLARIFY | REWRITE) with REWRITE → RETRIEVE → GENERATE → VALIDATE → EXECUTE →
(EXPLAIN | CORRECT → loop | CLARIFY).
Defines conditional edge routing based on the guard intent, validation
errors, execution errors, and retry count. Both the validation and execution
failure paths exit to CLARIFY once retries are exhausted (no dead loop), and
an ambiguous question routes straight to CLARIFY for a follow-up question
(kill-switch: clarify_ambiguity_enabled).

EXPLAIN itself is conditional: when its answerability verdict is "retry"
(answer_failure_reason set and the shared quota not exhausted) the graph
loops back to CORRECT for a semantic-failure retry — see
docs/DESIGN_explain_answerability.md. Termination is LLM-autonomous
(verdict answer / give_up both end the run).
"""

import logging

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.chat import chat
from app.agent.nodes.clarify import clarify
from app.agent.nodes.correct import correct
from app.agent.nodes.execute import execute
from app.agent.nodes.explain import explain
from app.agent.nodes.generate import generate
from app.agent.nodes.guard import guard
from app.agent.nodes.reject import reject
from app.agent.nodes.retrieve import retrieve
from app.agent.nodes.rewrite import rewrite
from app.agent.nodes.validate import validate
from app.agent.state import AgentState
from app.checkpoint import get_checkpointer
from app.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def route_after_guard(state: AgentState) -> str:
    """Route smalltalk to CHAT, unsafe to REJECT, ambiguous (when enabled) to
    CLARIFY, everything else to REWRITE."""
    intent = state.get("intent") or "data_query"
    if intent == "smalltalk":
        return "chat"
    if intent == "unsafe":
        return "reject"
    if intent == "ambiguous" and get_settings().clarify_ambiguity_enabled:
        return "clarify"
    return "rewrite"


def route_after_validate(state: AgentState) -> str:
    """Route to EXECUTE if validation passed; to CLARIFY if validation failed
    and retries are exhausted; to CORRECT otherwise."""
    if state.get("validation_error"):
        if state.get("retry_count", 0) >= get_settings().max_retries:
            return "clarify"
        return "correct"
    return "execute"


def route_after_execute(state: AgentState) -> str:
    """Route to EXPLAIN on success, CLARIFY if retries exhausted, CORRECT otherwise."""
    if state.get("execution_error") is None:
        return "explain"
    settings = get_settings()
    if state.get("retry_count", 0) >= settings.max_retries:
        return "clarify"
    return "correct"


def route_after_correct(state: AgentState) -> str:
    """After CORRECT, always go back through VALIDATE → EXECUTE loop."""
    return "validate"


def route_after_explain(state: AgentState) -> str:
    """EXPLAIN is terminal unless it explicitly requested a semantic retry
    (answerability verdict=retry with the shared quota not yet exhausted)."""
    if state.get("answer_failure_reason") is None:
        return END
    if state.get("retry_count", 0) >= get_settings().max_retries:
        return END  # defensive: final-attempt EXPLAIN must have cleared the reason
    return "correct"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Entry: GUARD classifies intent. smalltalk ends in CHAT (direct answer),
    unsafe ends in REJECT (static refusal), ambiguous ends in CLARIFY
    (follow-up question; falls back to the query pipeline when disabled),
    data_query goes through REWRITE (self-contained retrieval question) into
    the SQL pipeline:
    RETRIEVE → GENERATE → VALIDATE → EXECUTE → EXPLAIN | CORRECT | CLARIFY.
    CORRECT loops back to VALIDATE so each corrected SQL is re-validated
    before execution; both VALIDATE and EXECUTE route to CLARIFY once
    retry_count reaches max_retries. EXPLAIN ends the run on its answer /
    give_up verdicts and loops back to CORRECT only on a semantic-retry
    verdict (answer_failure_reason set, quota not exhausted).

    Returns:
        A compiled LangGraph runnable ready for .invoke() calls. State is
        checkpointed per thread — callers pass thread_id = session_id and
        the checkpointer restores chat memory across requests and restarts.
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("guard", guard)
    builder.add_node("chat", chat)
    builder.add_node("reject", reject)
    builder.add_node("rewrite", rewrite)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("validate", validate)
    builder.add_node("execute", execute)
    builder.add_node("correct", correct)
    builder.add_node("clarify", clarify)
    builder.add_node("explain", explain)

    # Linear edges
    builder.add_edge(START, "guard")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "validate")

    # Conditional edges
    builder.add_conditional_edges(
        "guard",
        route_after_guard,
        {"rewrite": "rewrite", "chat": "chat", "reject": "reject", "clarify": "clarify"},
    )
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {"execute": "execute", "correct": "correct", "clarify": "clarify"},
    )
    builder.add_conditional_edges(
        "execute",
        route_after_execute,
        {"explain": "explain", "correct": "correct", "clarify": "clarify"},
    )
    builder.add_conditional_edges(
        "correct",
        route_after_correct,
        {"validate": "validate"},
    )
    builder.add_conditional_edges(
        "explain",
        route_after_explain,
        {"correct": "correct", END: END},
    )

    # Terminal edges
    builder.add_edge("chat", END)
    builder.add_edge("reject", END)
    builder.add_edge("clarify", END)

    return builder.compile(checkpointer=get_checkpointer())


# Module-level compiled graph — import this for .invoke() calls
graph = build_graph()
