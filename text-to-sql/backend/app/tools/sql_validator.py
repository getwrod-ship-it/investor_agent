"""
app/tools/sql_validator.py

SQL syntax validator using sqlglot.
validate_sql(sql) parses the query without executing it and returns a
ValidationResult. Rejects queries containing FORBIDDEN_KEYWORDS before
even attempting to parse.

The query must parse under the MySQL dialect (the only supported business
database), and exactly one statement is allowed per request.
"""

import logging
from dataclasses import dataclass

import sqlglot
import sqlglot.errors

logger = logging.getLogger(__name__)

# Any SQL that modifies data or schema is unconditionally rejected
FORBIDDEN_KEYWORDS: frozenset[str] = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "TRUNCATE",
        "ALTER",
        "CREATE",
        "GRANT",
        "REVOKE",
    }
)

# Dialect accepted by the syntax check — MySQL is the only supported backend
_DIALECTS = ("mysql",)


@dataclass
class ValidationResult:
    """Outcome of a SQL validation check."""

    is_valid: bool
    error_message: str | None = None


def validate_sql(sql: str) -> ValidationResult:
    """
    Check SQL for forbidden keywords and syntax errors without executing it.

    Checks are applied in order:
      1. Forbidden keyword scan (case-insensitive, whole-token match via sqlglot tokeniser)
      2. sqlglot parse against the MySQL dialect
      3. Exactly one SQL statement is allowed

    Args:
        sql: The SQL string to validate.

    Returns:
        ValidationResult with is_valid=True on success, or is_valid=False and
        a human-readable error_message on failure.
    """
    if not sql or not sql.strip():
        return ValidationResult(is_valid=False, error_message="SQL is empty")

    # --- Forbidden keyword check ---
    try:
        tokens = sqlglot.tokenize(sql)
        token_values = {t.text.upper() for t in tokens}
    except sqlglot.errors.TokenError as exc:
        return ValidationResult(is_valid=False, error_message=f"Tokenisation error: {exc}")

    hit = token_values & FORBIDDEN_KEYWORDS
    if hit:
        keyword = sorted(hit)[0]
        logger.warning("validate_sql: forbidden keyword '%s' found in SQL", keyword)
        return ValidationResult(
            is_valid=False,
            error_message=f"Forbidden keyword '{keyword}' is not allowed in read-only mode",
        )

    # --- Syntax check (MySQL dialect) ---
    statements: list | None = None
    parse_errors: list[str] = []
    for dialect in _DIALECTS:
        try:
            statements = sqlglot.parse(sql, dialect=dialect, error_level=sqlglot.ErrorLevel.RAISE)
            break
        except sqlglot.errors.ParseError as exc:
            parse_errors.append(f"{dialect}: {exc}")

    if statements is None:
        logger.debug("validate_sql: parse errors: %s", parse_errors)
        return ValidationResult(
            is_valid=False,
            error_message=f"SQL syntax error: {parse_errors[-1] if parse_errors else 'unparseable'}",
        )

    if len(statements) > 1:
        return ValidationResult(
            is_valid=False,
            error_message="Multiple SQL statements are not allowed",
        )

    if not statements:
        return ValidationResult(is_valid=False, error_message="No SQL statement found")

    logger.debug("validate_sql: OK")
    return ValidationResult(is_valid=True)
