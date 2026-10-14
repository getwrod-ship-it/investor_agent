"""
app/agent/state.py

AgentState TypedDict — the single shared state object threaded through all
7 LangGraph nodes. Fields cover the full agent lifecycle: question intake,
RAG context, SQL generation, validation, execution, self-correction, and
final explanation.
"""

from typing import Any, Optional
from typing_extensions import TypedDict


class CorrectionRecord(TypedDict):
    """One entry in the correction history — records a failed attempt."""

    attempt_number: int
    failed_sql: str
    error_message: str
    corrected_sql: Optional[str]


class ExecutionResult(TypedDict):
    """Structured result returned by the EXECUTE node on success."""

    rows: list[dict[str, Any]]
    row_count: int
    execution_time_ms: float


class ChatTurn(TypedDict):
    """One prior turn in the same session, loaded by the route layer."""

    question: str
    sql: Optional[str]        # smalltalk turns have no SQL
    answer: str               # final_explanation, truncated for token budget


class AgentState(TypedDict):
    """
    Shared state passed between every LangGraph node.

    Populated incrementally as the agent moves through the graph:
      RETRIEVE  → retrieved_schema, retrieved_docs
      GENERATE  → generated_sql
      VALIDATE  → validation_error
      EXECUTE   → execution_result | execution_error
      CORRECT   → retry_count, correction_history
      CLARIFY   → needs_clarification, clarification_message, clarify_reason
      EXPLAIN   → final_explanation, answer_failure_reason
    """

    # Input
    question: str
    session_id: str

    # Intent guard (set by GUARD)
    intent: str                            # "data_query" | "smalltalk" | "unsafe"
    direct_answer: Optional[str]           # smalltalk reply or unsafe refusal
    rewritten_question: Optional[str]      # self-contained question for retrieval

    # Session short-term memory (maintained by the route layer via the
    # checkpointer, read-only for nodes)
    chat_history: list[ChatTurn]
    history_summary: Optional[str]   # rolling summary of turns older than the window
    memory_turn_count: int           # turns folded into memory (window + summary) — self-heal watermark

    # Long-term memory (cross-session, set by the route layer via mem0 recall,
    # read-only for nodes — see DESIGN_long_term_memory.md). Both hold already
    # formatted lines and are reset explicitly on every turn like all other
    # per-turn fields.
    long_term_facts: list[str]       # current-valid facts: identity / metric conventions / preferences
    episodic_memories: list[str]     # episodic excerpts: "when was what asked, what was concluded"

    # RAG context (set by RETRIEVE)
    retrieved_schema: list[str]   # schema chunk texts, highest-score first
    retrieved_docs: list[str]     # doc chunk texts, highest-score first

    # SQL generation (set by GENERATE)
    generated_sql: Optional[str]
    sql_reasoning: Optional[str]
    sql_assumptions: list[str]
    sql_confidence: float          # 0.0–1.0

    # Validation (set by VALIDATE)
    validation_error: Optional[str]

    # Execution (set by EXECUTE)
    execution_result: Optional[ExecutionResult]
    execution_error: Optional[str]

    # Self-correction (set by CORRECT)
    retry_count: int
    correction_history: list[CorrectionRecord]

    # Clarification (set by CLARIFY)
    needs_clarification: bool
    clarification_message: Optional[str]
    clarify_reason: Optional[str]        # "ambiguous" | "retry_exhausted" | "infra_failure"

    # Final output (set by EXPLAIN)
    final_explanation: Optional[str]
    answer_failure_reason: Optional[str]    # set by EXPLAIN (answerability gate):
                                           # why the result cannot answer the
                                           # question AND a different SQL strategy
                                           # is worth trying (verdict="retry"
                                           # only; "give_up" goes straight to
                                           # final_explanation). Consumed and
                                           # cleared by CORRECT.

    # Observability — correlation id for this run (also persisted with QueryRun)
    trace_id: Optional[str]
