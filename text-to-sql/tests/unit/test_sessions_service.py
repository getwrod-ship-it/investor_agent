"""
ChatSession list-entry service: lazy upsert (crash-consistent turn_count),
list ordering, chronological message fetch (incl. unsafe), cascade delete.
"""

from datetime import datetime, timedelta, timezone

from app.models.db import ChatSession, CorrectionAttempt, QueryRun
from app.services.sessions import (
    delete_chat_session,
    get_session_messages,
    list_chat_sessions,
    upsert_chat_session,
)
from sqlmodel import select

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _add_run(db, session_id: str, i: int, *, intent: str = "data_query",
             created: datetime | None = None) -> QueryRun:
    run = QueryRun(
        session_id=session_id,
        question=f"问题{i}",
        intent=intent,
        generated_sql=None if intent == "smalltalk" else f"SELECT {i}",
        final_explanation=f"答案{i}",
        created_at=created or _BASE + timedelta(minutes=i),
    )
    db.add(run)
    return run


def test_upsert_creates_entry_with_truncated_title(db_session):
    long_question = "这是一个特别特别特别特别特别特别长的问题标题内容"
    _add_run(db_session, "s1", 1)
    db_session.commit()

    row = upsert_chat_session(db_session, session_id="s1", question=long_question)

    assert len(row.title) == 20
    assert row.turn_count == 1
    assert row.last_question == long_question


def test_upsert_recomputes_turn_count_after_missed_upsert(db_session):
    # crash scenario: two audit rows landed but no chat_session row exists
    _add_run(db_session, "s2", 1)
    _add_run(db_session, "s2", 2)
    db_session.commit()

    row = upsert_chat_session(db_session, session_id="s2", question="问题2")

    assert row.turn_count == 2  # recomputed from query_run, not incremented


def test_upsert_keeps_first_question_title(db_session):
    _add_run(db_session, "s3", 1)
    db_session.commit()
    upsert_chat_session(db_session, session_id="s3", question="问题1")

    _add_run(db_session, "s3", 2)
    db_session.commit()
    row = upsert_chat_session(db_session, session_id="s3", question="问题2")

    assert row.title == "问题1"  # frozen at the first question
    assert row.turn_count == 2
    assert row.last_question == "问题2"


def test_list_orders_by_updated_at_desc(db_session):
    for sid, minutes in (("older", 1), ("newer", 5)):
        db_session.add(
            ChatSession(
                id=sid,
                title=sid,
                created_at=_BASE,
                updated_at=_BASE + timedelta(minutes=minutes),
            )
        )
    db_session.commit()

    assert [r.id for r in list_chat_sessions(db_session)] == ["newer", "older"]


def test_messages_chronological_including_unsafe(db_session):
    _add_run(db_session, "s4", 1, intent="data_query")
    _add_run(db_session, "s4", 2, intent="unsafe")
    _add_run(db_session, "s4", 3, intent="smalltalk")
    db_session.commit()

    msgs = get_session_messages(db_session, "s4")
    assert [m.question for m in msgs] == ["问题1", "问题2", "问题3"]


def test_delete_cascades_to_attempts_and_runs(db_session):
    run = _add_run(db_session, "s5", 1)
    db_session.commit()
    db_session.add(
        CorrectionAttempt(
            query_run_id=run.id,
            attempt_number=1,
            failed_sql="SELECT 1",
            error_message="boom",
        )
    )
    db_session.add(ChatSession(id="s5", title="t"))
    db_session.commit()

    assert delete_chat_session(db_session, "s5") is True
    assert db_session.exec(select(QueryRun)).all() == []
    assert db_session.exec(select(CorrectionAttempt)).all() == []
    assert db_session.get(ChatSession, "s5") is None


def test_delete_unknown_session_returns_false(db_session):
    assert delete_chat_session(db_session, "nope") is False
