"""
frontend/app.py

Streamlit chat interface for the Investor Agent (Text-to-SQL).
Sends questions to POST /query, displays SQL, results, and explanations.
Sidebar shows the persisted session list (open/continue/delete past
conversations across restarts), connection status, a schema browser, and
last 5 queries.
"""

import os
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# 8888 because Hyper-V/WSL reserves 7979-8778 on this machine (8000 included)
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8888")
REQUEST_TIMEOUT = 60  # seconds


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _post_query(question: str, session_id: str) -> dict[str, Any]:
    """Call POST /query and return the parsed JSON response."""
    resp = requests.post(
        f"{API_BASE}/query",
        json={"question": question, "session_id": session_id},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _get_schema_status() -> dict[str, Any] | None:
    """Call GET /schema/status and return the parsed JSON, or None on error."""
    try:
        resp = requests.get(f"{API_BASE}/schema/status", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _get_schema_tables() -> list[dict[str, Any]]:
    """Call GET /schema/tables and return the live table metadata, or []."""
    try:
        resp = requests.get(f"{API_BASE}/schema/tables", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def _get_history(limit: int = 5) -> list[dict[str, Any]]:
    """Call GET /history and return recent query runs."""
    try:
        resp = requests.get(f"{API_BASE}/history", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def _get_sessions(limit: int = 20) -> list[dict[str, Any]]:
    """Call GET /sessions and return the persisted session list, or []."""
    try:
        resp = requests.get(f"{API_BASE}/sessions", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def _get_session_detail(session_id: str) -> dict[str, Any] | None:
    """Call GET /sessions/{id} and return the full replay, or None on error."""
    try:
        resp = requests.get(f"{API_BASE}/sessions/{session_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _delete_session(session_id: str) -> bool:
    """Call DELETE /sessions/{id}; return True on success."""
    try:
        resp = requests.delete(f"{API_BASE}/sessions/{session_id}", timeout=10)
        return resp.status_code == 204
    except Exception:
        return False


def _health_check() -> bool:
    """Return True if the backend is reachable."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Session management helpers
# ---------------------------------------------------------------------------


def _switch_session(session_id: str, messages: list[dict[str, Any]]) -> None:
    """Switch to another (or a brand-new) session and rerun to re-render."""
    st.session_state.session_id = session_id
    st.session_state.messages = messages
    st.session_state.awaiting_clarification = False
    st.rerun()


def _detail_to_messages(detail: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a session detail onto the chat replay format.

    Each replayed turn is a ChatTurnView whose field shape matches
    QueryResponse, so _render_assistant_message renders replayed turns
    identically to live replies — SQL expanders, result tables, smalltalk
    replies and refusals included.
    """
    messages: list[dict[str, Any]] = []
    for turn in detail.get("messages", []):
        messages.append({"role": "user", "content": turn["question"], "data": {}})
        messages.append({"role": "assistant", "content": "", "data": turn})
    return messages


def _relative_time(iso: str) -> str:
    """Render an ISO timestamp as a short relative time (e.g. 3 小时前)."""
    try:
        ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return ""
    seconds = (datetime.now(ts.tzinfo) - ts).total_seconds()
    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟前"
    if seconds < 86400:
        return f"{int(seconds // 3600)} 小时前"
    return f"{int(seconds // 86400)} 天前"


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def _init_session() -> None:
    """Initialise Streamlit session state on first run."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role", "content", "data"}
    if "awaiting_clarification" not in st.session_state:
        st.session_state.awaiting_clarification = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    """Render the sidebar: sessions, connection status, schema, recent queries."""
    with st.sidebar:
        st.title("Investor Agent")

        # --- Sessions (persisted server-side; open to replay & continue) ---
        st.subheader("💬 会话")
        if st.button("➕ 新建会话", use_container_width=True):
            # local uuid only — the backend lazily creates the row on the
            # first question, so empty sessions never enter the list
            _switch_session(str(uuid.uuid4()), messages=[])
        sessions = _get_sessions()
        if sessions:
            for s in sessions:
                active = s["session_id"] == st.session_state.session_id
                col_btn, col_del = st.columns([5, 1])
                if col_btn.button(
                    ("📍 " if active else "") + s["title"],
                    key=f"sess-{s['session_id']}",
                    use_container_width=True,
                ):
                    detail = _get_session_detail(s["session_id"])
                    _switch_session(
                        s["session_id"],
                        messages=_detail_to_messages(detail) if detail else [],
                    )
                st.caption(
                    f"{s['turn_count']} 轮 · {_relative_time(s['updated_at'])}"
                )
                if col_del.button("🗑", key=f"del-{s['session_id']}", help="删除会话"):
                    _delete_session(s["session_id"])
                    if s["session_id"] == st.session_state.session_id:
                        _switch_session(str(uuid.uuid4()), messages=[])
                    else:
                        st.rerun()
        else:
            st.caption("暂无历史会话")

        st.divider()

        # --- Connection status ---
        st.subheader("Status")
        if _health_check():
            st.success("Backend connected", icon="✅")
        else:
            st.error("Backend unreachable", icon="🔴")

        st.divider()

        # --- Schema browser ---
        st.subheader("Schema Browser")
        status = _get_schema_status()
        if status:
            counts = " · ".join(
                f"{c['name']}: {c['vector_count']}" for c in status.get("collections", [])
            )
            st.caption(f"Vector store — {counts}")
            if status.get("last_snapshot_at"):
                st.caption(f"Last refresh: {status['last_snapshot_at'][:19]}")
        else:
            st.warning("Schema status unavailable")

        # Live table reference fetched from the business database
        with st.expander("Tables & columns", expanded=False):
            tables = _get_schema_tables()
            if tables:
                for table in tables:
                    st.markdown(f"**{table['table_name']}**")
                    st.caption(", ".join(col["name"] for col in table["columns"]))
            else:
                st.caption(
                    "No tables found. Check that BUSINESS_DATABASE_URL is correct, "
                    "then run `make embed` or POST /schema/refresh."
                )

        st.divider()

        # --- Recent queries ---
        st.subheader("Recent Queries")
        history = _get_history(limit=5)
        if history:
            for run in history:
                icon = "✅" if run.get("succeeded") else "❌"
                label = run["question"][:50] + ("…" if len(run["question"]) > 50 else "")
                st.caption(f"{icon} {label}")
        else:
            st.caption("No queries yet")


# ---------------------------------------------------------------------------
# Chat rendering helpers
# ---------------------------------------------------------------------------


def _render_assistant_message(data: dict[str, Any]) -> None:
    """Render a complete assistant response bubble."""
    intent = data.get("intent", "data_query")
    needs_clarification = data.get("needs_clarification", False)
    succeeded = data.get("succeeded", False)
    retry_count = data.get("retry_count", 0)
    sql = data.get("sql")
    result = data.get("result", [])
    explanation = data.get("explanation")
    clarification_message = data.get("clarification_message")
    direct_answer = data.get("direct_answer")
    rewritten_question = data.get("rewritten_question")

    # Intent branches: smalltalk / unsafe never reached the SQL pipeline
    if intent == "smalltalk":
        st.markdown(direct_answer or explanation or "（闲聊回答缺失）")
        return
    if intent == "unsafe":
        st.warning(direct_answer or "这个请求我不能处理：本系统只支持只读查询。")
        return

    # Retry badge
    if retry_count > 0:
        st.info(f"Self-corrected after {retry_count} attempt(s)", icon="🔄")

    if needs_clarification:
        st.warning(clarification_message or "I couldn't generate a valid query. Please rephrase.")
        return

    if not succeeded:
        st.error("The agent was unable to answer this question.")
        return

    # Rewrite transparency: show what the system understood a follow-up to mean
    if rewritten_question and rewritten_question != data.get("question"):
        st.caption(f"系统理解为：{rewritten_question}")

    # SQL block
    if sql:
        with st.expander("Generated SQL", expanded=True):
            st.code(sql, language="sql")

    # Results dataframe
    if result:
        df = pd.DataFrame(result)
        st.dataframe(df, use_container_width=True)
        st.caption(f"{len(df):,} row(s) returned")
    else:
        st.caption("Query returned no rows.")

    # Plain-language explanation
    if explanation:
        st.markdown(explanation)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Investor Agent",
        page_icon="🔎",
        layout="wide",
    )

    _init_session()
    _render_sidebar()

    st.title("🔎 Investor Agent")
    st.caption("Ask questions about your business database in plain language.")

    # Replay chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_assistant_message(msg["data"])

    # Chat input
    prompt = st.chat_input(
        "e.g. Show the 10 most recent records from the largest table",
        disabled=False,
    )

    if prompt:
        # Append and render user message
        st.session_state.messages.append({"role": "user", "content": prompt, "data": {}})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call the agent
        with st.chat_message("assistant"):
            with st.spinner("Generating SQL…"):
                try:
                    data = _post_query(prompt, st.session_state.session_id)
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the backend. Is `docker compose up` running?")
                    st.stop()
                except requests.exceptions.HTTPError as exc:
                    st.error(f"Backend error: {exc.response.status_code} — {exc.response.text[:200]}")
                    st.stop()
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
                    st.stop()

            _render_assistant_message(data)

        # Persist assistant message for history replay
        st.session_state.messages.append({"role": "assistant", "content": "", "data": data})

        # If clarification needed, allow follow-up in next turn
        if data.get("needs_clarification"):
            st.session_state.awaiting_clarification = True
        else:
            st.session_state.awaiting_clarification = False


if __name__ == "__main__":
    main()
