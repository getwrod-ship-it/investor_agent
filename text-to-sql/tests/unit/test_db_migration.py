"""
SQLite migration: the `intent` and `clarify_reason` columns are added to
pre-existing query_run tables (create_all never alters existing tables).
"""

from app.db import _ensure_query_run_clarify_reason_column, _ensure_query_run_intent_column
from sqlalchemy import create_engine, inspect


def test_adds_intent_column_to_legacy_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path.as_posix()}/legacy.db")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE query_run ("
            "id VARCHAR PRIMARY KEY, session_id VARCHAR, question VARCHAR, "
            "created_at TIMESTAMP)"
        )

    _ensure_query_run_intent_column(engine)

    columns = {c["name"] for c in inspect(engine).get_columns("query_run")}
    assert "intent" in columns


def test_noop_when_column_already_exists(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path.as_posix()}/current.db")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE query_run ("
            "id VARCHAR PRIMARY KEY, session_id VARCHAR, question VARCHAR, "
            "intent VARCHAR, created_at TIMESTAMP)"
        )

    _ensure_query_run_intent_column(engine)  # must not raise

    columns = {c["name"] for c in inspect(engine).get_columns("query_run")}
    assert "intent" in columns


def test_noop_when_table_missing(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path.as_posix()}/empty.db")

    _ensure_query_run_intent_column(engine)  # must not raise or create anything

    assert not inspect(engine).has_table("query_run")


def test_adds_clarify_reason_column_to_legacy_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path.as_posix()}/legacy.db")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE query_run ("
            "id VARCHAR PRIMARY KEY, session_id VARCHAR, question VARCHAR, "
            "intent VARCHAR, created_at TIMESTAMP)"
        )

    _ensure_query_run_clarify_reason_column(engine)

    columns = {c["name"] for c in inspect(engine).get_columns("query_run")}
    assert "clarify_reason" in columns


def test_clarify_reason_noop_when_column_already_exists(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path.as_posix()}/current.db")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE query_run ("
            "id VARCHAR PRIMARY KEY, session_id VARCHAR, question VARCHAR, "
            "intent VARCHAR, clarify_reason VARCHAR, created_at TIMESTAMP)"
        )

    _ensure_query_run_clarify_reason_column(engine)  # must not raise

    columns = {c["name"] for c in inspect(engine).get_columns("query_run")}
    assert "clarify_reason" in columns


def test_clarify_reason_noop_when_table_missing(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path.as_posix()}/empty.db")

    _ensure_query_run_clarify_reason_column(engine)  # must not raise or create anything

    assert not inspect(engine).has_table("query_run")
