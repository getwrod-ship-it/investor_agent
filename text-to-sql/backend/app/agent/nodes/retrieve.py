"""
app/agent/nodes/retrieve.py

RETRIEVE node: embeds the user question and queries both Qdrant collections
(text2sql_schema and text2sql_docs) in parallel. Merges results by
relevance score and populates retrieved_schema and retrieved_docs in AgentState.
"""

import logging

from app.agent.state import AgentState
from app.rag.retriever import retrieve_context

logger = logging.getLogger(__name__)

# Chunks fetched per collection per question
_TOP_K = 3


def retrieve(state: AgentState) -> AgentState:
    """
    Query both Qdrant collections for context relevant to the user question.

    Calls retrieve_context() which searches text2sql_schema and text2sql_docs
    concurrently. Schema chunks are split into retrieved_schema; doc chunks into
    retrieved_docs. Both lists are ordered by descending relevance score.

    Retrieval uses the REWRITE node's self-contained question when available
    (elliptical follow-ups like "那前个月呢" would otherwise recall garbage);
    falls back to the original question.

    Args:
        state: Current AgentState — must have 'question' set.

    Returns:
        Updated AgentState with retrieved_schema and retrieved_docs populated.
    """
    original_question = state["question"]
    question = state.get("rewritten_question") or original_question
    chunks = retrieve_context(question, top_k=_TOP_K)

    schema_texts = [c.text for c in chunks if c.source == "schema"]
    doc_texts = [c.text for c in chunks if c.source == "docs"]

    logger.info(
        "RETRIEVE: question=%r → %d schema chunk(s), %d doc chunk(s)",
        question[:60],
        len(schema_texts),
        len(doc_texts),
    )

    return {
        **state,
        "retrieved_schema": schema_texts,
        "retrieved_docs": doc_texts,
    }
