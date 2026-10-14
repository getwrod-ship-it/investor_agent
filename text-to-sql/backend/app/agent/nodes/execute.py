"""
app/agent/nodes/execute.py

EXECUTE node: runs the validated SQL against the business database via
mysql_executor, enforcing forbidden-keyword checks, a configurable timeout,
and a row limit. Populates execution_result or execution_error in AgentState.
"""

import logging

from app.agent.state import AgentState, ExecutionResult
from app.tools.mysql_executor import ForbiddenQueryError, QueryExecutionError, execute_query

logger = logging.getLogger(__name__)


def execute(state: AgentState) -> AgentState:
    """
    Execute the validated SQL against the business database.

    Calls mysql_executor.execute_query() which enforces:
      - Forbidden keyword check (raises ForbiddenQueryError)
      - Statement timeout where the dialect supports one
      - Row limit capped at settings.query_row_limit

    On success: populates execution_result and clears execution_error.
    On failure: populates execution_error and clears execution_result.

    Args:
        state: Current AgentState — must have generated_sql set (and validated).

    Returns:
        Updated AgentState with execution_result or execution_error populated.
    """
    sql = state.get("generated_sql", "")

    if not sql:
        return {**state, "execution_result": None, "execution_error": "No SQL to execute"}

    try:
        result = execute_query(sql)
        logger.info("EXECUTE: %d row(s) in %.1f ms", result.row_count, result.execution_time_ms)
        execution_result: ExecutionResult = {
            "rows": result.rows,
            "row_count": result.row_count,
            "execution_time_ms": result.execution_time_ms,
        }
        return {**state, "execution_result": execution_result, "execution_error": None}

    except ForbiddenQueryError as exc:
        logger.warning("EXECUTE: forbidden query blocked — %s", exc)
        return {**state, "execution_result": None, "execution_error": str(exc)}

    except QueryExecutionError as exc:
        logger.warning("EXECUTE: query error — %s", exc)
        return {**state, "execution_result": None, "execution_error": str(exc)}

    except Exception as exc:
        logger.error("EXECUTE: unexpected error — %s", exc)
        return {**state, "execution_result": None, "execution_error": f"Unexpected error: {exc}"}
