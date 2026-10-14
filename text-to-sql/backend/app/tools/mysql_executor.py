"""
app/tools/mysql_executor.py

Read-only query executor for the MySQL business database.
execute_query(sql) enforces forbidden-keyword checks, applies a
max_execution_time statement timeout, and caps results at query_row_limit rows.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError

from app.business_db import get_business_engine
from app.config import get_settings
from app.tools.sql_validator import FORBIDDEN_KEYWORDS

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structured result from a successful query execution."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0


class ForbiddenQueryError(Exception):
    """Raised when the SQL contains a forbidden keyword."""


class QueryExecutionError(Exception):
    """Raised when the database returns an error during execution."""


def execute_query(sql: str) -> QueryResult:
    """
    Execute a read-only SQL query against the MySQL business database.

    Safety checks applied before execution:
      1. Forbidden keyword scan — raises ForbiddenQueryError immediately.
      2. Statement timeout via SET SESSION max_execution_time.
      3. Results capped at settings.query_row_limit rows.

    Args:
        sql: A SELECT (or equivalent read-only) SQL string.

    Returns:
        QueryResult with rows (list of dicts), row_count, and execution_time_ms.

    Raises:
        ForbiddenQueryError: If the SQL contains a forbidden keyword.
        QueryExecutionError: If the database returns an error.
    """
    settings = get_settings()

    # --- Forbidden keyword guard ---
    upper_tokens = {token.upper() for token in sql.split()}
    hit = upper_tokens & FORBIDDEN_KEYWORDS
    if hit:
        keyword = sorted(hit)[0]
        raise ForbiddenQueryError(
            f"Forbidden keyword '{keyword}' is not permitted in read-only mode"
        )

    engine = get_business_engine()
    timeout_ms = settings.query_timeout_seconds * 1000
    row_limit = settings.query_row_limit

    try:
        with engine.connect() as conn:
            # Applies to this fresh connection; NullPool discards it afterwards
            conn.execute(text(f"SET SESSION max_execution_time = {int(timeout_ms)}"))

            start = time.perf_counter()
            result = conn.execute(text(sql))
            rows = [dict(row._mapping) for row in result.fetchmany(row_limit)]
            elapsed_ms = (time.perf_counter() - start) * 1000
            conn.commit()

        logger.info(
            "execute_query (mysql): %d row(s) in %.1f ms (limit=%d)",
            len(rows),
            elapsed_ms,
            row_limit,
        )
        return QueryResult(rows=rows, row_count=len(rows), execution_time_ms=elapsed_ms)

    except (OperationalError, DBAPIError) as exc:
        orig = str(getattr(exc, "orig", exc))
        if _looks_like_timeout(orig):
            raise QueryExecutionError(
                f"Query exceeded {settings.query_timeout_seconds}s timeout"
            ) from exc
        raise QueryExecutionError(orig) from exc
    except Exception as exc:
        raise QueryExecutionError(str(exc)) from exc


def _looks_like_timeout(message: str) -> bool:
    """Heuristically detect a MySQL max_execution_time timeout (error 3024)."""
    lowered = message.lower()
    if "max_execution_time" in lowered and "exceeded" in lowered:
        return True
    return any(
        marker in lowered
        for marker in (
            "maximum statement execution time exceeded",
            "query execution was interrupted",
        )
    )
