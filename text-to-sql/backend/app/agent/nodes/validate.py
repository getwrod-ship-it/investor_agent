"""
app/agent/nodes/validate.py

VALIDATE node: syntax-checks the generated SQL with sqlglot before it touches
the business database. On failure, sets validation_error so the graph routes
to CORRECT.
"""

import logging

from app.agent.state import AgentState
from app.tools.sql_validator import validate_sql

logger = logging.getLogger(__name__)


def validate(state: AgentState) -> AgentState:
    """
    Syntax-check the generated SQL using sqlglot.

    If generated_sql is None (e.g. generation failed) the node immediately
    sets a validation_error so the graph routes to CORRECT.

    Args:
        state: Current AgentState — must have generated_sql set by GENERATE.

    Returns:
        Updated AgentState with validation_error=None on success, or
        validation_error set to a human-readable message on failure.
    """
    sql = state.get("generated_sql")

    if not sql:
        logger.warning("VALIDATE: no SQL was generated")
        return {**state, "validation_error": "No SQL was generated"}

    result = validate_sql(sql)

    if result.is_valid:
        logger.info("VALIDATE: OK — sql_preview=%r", sql[:80])
        return {**state, "validation_error": None}

    logger.warning("VALIDATE: failed — %s", result.error_message)
    return {**state, "validation_error": result.error_message}
