"""
mcp_server/server.py

MCP server exposing three tools to MCP clients (e.g. Claude Code):
  - query_tool:   run a natural-language question through the Text-to-SQL agent
  - schema_tool:  look up tables/columns matching a keyword
  - history_tool: fetch recent query runs from the agent history

Run with:  python mcp_server/server.py
Or via Docker: the mcp_server service in docker-compose.yml
"""

import os
import uuid
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# 8888 because Hyper-V/WSL reserves 7979-8778 on this machine (8000 included)
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8888")
REQUEST_TIMEOUT = 60

mcp = FastMCP(
    name="text2sql-agent",
    instructions=(
        "Tools for querying the connected business database in plain language. "
        "Use query_tool to ask data questions, schema_tool to explore table "
        "structure, and history_tool to review recent queries."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_tables() -> list[dict[str, Any]]:
    """
    Fetch the live table/column metadata from the backend GET /schema/tables.

    Returns a list of {"table_name": str, "columns": [...], "foreign_keys": [...]}
    dicts, or an empty list when the backend is unreachable.
    """
    try:
        resp = requests.get(f"{API_BASE}/schema/tables", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def query_tool(question: str, session_id: str = "") -> str:
    """
    Run a natural-language question against the connected business database.

    The agent generates SQL, executes it read-only, and returns results with a
    plain-language explanation.

    Args:
        question:   The question in plain language (5–500 characters).
        session_id: Optional session identifier for conversation continuity.
                    A UUID is generated automatically if not provided.

    Returns:
        A formatted string containing the generated SQL, result preview,
        and plain-language explanation. On failure, returns an error message
        or a clarification request.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        resp = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "session_id": session_id},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
    except requests.exceptions.ConnectionError:
        return "Error: Cannot reach the backend API. Ensure docker-compose is running."
    except requests.exceptions.HTTPError as exc:
        return f"Error: Backend returned {exc.response.status_code}: {exc.response.text[:300]}"
    except Exception as exc:
        return f"Error: {exc}"

    # Intent branches: smalltalk / unsafe never reached the SQL pipeline
    intent = data.get("intent", "data_query")
    if intent == "smalltalk":
        return data.get("direct_answer") or data.get("explanation") or ""
    if intent == "unsafe":
        return data.get("direct_answer") or (
            "Request refused: this system only performs read-only queries."
        )

    if data.get("needs_clarification"):
        return f"Clarification needed:\n{data.get('clarification_message', '')}"

    if not data.get("succeeded"):
        return "The agent could not generate a valid query for this question."

    lines: list[str] = []

    rewritten_question = data.get("rewritten_question")
    if rewritten_question and rewritten_question != data.get("question"):
        lines.append(f"(Interpreted as: {rewritten_question})")

    sql = data.get("sql", "")
    if sql:
        lines.append(f"SQL:\n```sql\n{sql}\n```")

    retry_count = data.get("retry_count", 0)
    if retry_count > 0:
        lines.append(f"(self-corrected after {retry_count} attempt(s))")

    result = data.get("result", [])
    row_count = data.get("row_count", 0)
    if result:
        # Show up to 10 rows as a compact table
        preview = result[:10]
        if preview:
            headers = list(preview[0].keys())
            lines.append(f"\nResults ({row_count} row(s)):")
            lines.append(" | ".join(headers))
            lines.append("-" * (len(" | ".join(headers))))
            for row in preview:
                lines.append(" | ".join(str(row.get(h, "")) for h in headers))
            if row_count > 10:
                lines.append(f"... ({row_count - 10} more rows)")
    else:
        lines.append("\nResults: (no rows returned)")

    explanation = data.get("explanation", "")
    if explanation:
        lines.append(f"\nExplanation:\n{explanation}")

    lines.append(f"\ntrace_id: {data.get('trace_id', 'n/a')}")

    return "\n".join(lines)


@mcp.tool()
def schema_tool(keyword: str) -> str:
    """
    Look up tables and columns in the business database schema matching a keyword.

    Searches live table names and column names for the given keyword
    (case-insensitive). Also returns live Qdrant collection stats from the
    backend.

    Args:
        keyword: A table name, column name, or topic (e.g. "orders", "user", "amount").

    Returns:
        Formatted text listing matching tables and their columns, plus
        live vector collection stats.
    """
    kw = keyword.lower()
    lines: list[str] = []

    tables = _fetch_tables()

    if not tables:
        lines.append(
            "Could not fetch the schema from the backend. "
            "Ensure the API is running and BUSINESS_DATABASE_URL is configured."
        )
        return "\n".join(lines)

    # Match tables and columns
    matches: list[dict[str, Any]] = []
    for table in tables:
        table_name = table["table_name"]
        col_matches = [
            c for c in table["columns"]
            if kw in c["name"].lower() or kw in table_name.lower()
        ]
        if col_matches:
            matches.append({"table_name": table_name, "columns": col_matches})

    if matches:
        lines.append(f"Schema matches for '{keyword}':\n")
        for table in matches:
            lines.append(f"Table: {table['table_name']}")
            for col in table["columns"]:
                pk = "  [primary key]" if col.get("is_primary_key") else ""
                lines.append(f"  {col['name']} ({col['data_type']}){pk}")
            lines.append("")
    else:
        lines.append(f"No schema matches for '{keyword}'.")
        lines.append(f"Available tables: {', '.join(t['table_name'] for t in tables)}")

    # Live collection stats
    try:
        resp = requests.get(f"{API_BASE}/schema/status", timeout=10)
        if resp.status_code == 200:
            status = resp.json()
            lines.append("\nLive Qdrant collection stats:")
            for col in status.get("collections", []):
                lines.append(f"  {col['name']}: {col['vector_count']} vectors")
            if status.get("last_snapshot_at"):
                lines.append(f"  Last refreshed: {status['last_snapshot_at']}")
    except Exception:
        lines.append("\n(Could not fetch live collection stats)")

    return "\n".join(lines)


@mcp.tool()
def history_tool(limit: int = 5) -> str:
    """
    Retrieve recent query runs from the agent history.

    Args:
        limit: Number of recent queries to return (1–20, default 5).

    Returns:
        Formatted list of recent queries with their SQL and outcome.
    """
    limit = max(1, min(limit, 20))

    try:
        resp = requests.get(
            f"{API_BASE}/history",
            params={"limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        runs: list[dict[str, Any]] = resp.json()
    except requests.exceptions.ConnectionError:
        return "Error: Cannot reach the backend API."
    except Exception as exc:
        return f"Error fetching history: {exc}"

    if not runs:
        return "No query history found."

    lines = [f"Last {len(runs)} query run(s):\n"]
    for i, run in enumerate(runs, 1):
        status = "✅" if run.get("succeeded") else ("❓" if run.get("needs_clarification") else "❌")
        lines.append(f"{i}. {status} {run['question']}")
        if run.get("generated_sql"):
            sql_preview = run["generated_sql"].replace("\n", " ")[:100]
            lines.append(f"   SQL: {sql_preview}{'…' if len(run['generated_sql']) > 100 else ''}")
        retries = run.get("retry_count", 0)
        if retries:
            lines.append(f"   Retries: {retries}")
        lines.append(f"   Created: {run.get('created_at', '')[:19]}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
