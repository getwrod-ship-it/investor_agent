"""
get_recent_session_turns: session filtering, ordering, window size,
unsafe exclusion, legacy NULL-intent inclusion; save_query_run persists intent.
"""

from datetime import UTC, datetime

from app.models.db import QueryRun
from app.services.history import get_recent_session_turns, save_query_run


def _turn(
    session_id: str,
    question: str,
    *,
    intent: str | None = "data_query",
    created_at: datetime,
    sql: str | None = "SELECT 1",
    explanation: str | None = "解释",
) -> QueryRun:
    return QueryRun(
        session_id=session_id,
        question=question,
        intent=intent,
        generated_sql=sql,
        final_explanation=explanation,
        created_at=created_at,
    )


def _at(minute: int) -> datetime:
    return datetime(2026, 10, 12, 10, minute, tzinfo=UTC)


def test_returns_turns_chronologically(db_session):
    db_session.add_all(
        [
            _turn("s1", "第一问", created_at=_at(0)),
            _turn("s1", "第二问", created_at=_at(1)),
            _turn("s1", "第三问", created_at=_at(2)),
        ]
    )
    db_session.commit()

    turns = get_recent_session_turns(db_session, "s1")

    assert [t.question for t in turns] == ["第一问", "第二问", "第三问"]


def test_filters_by_session(db_session):
    db_session.add_all(
        [
            _turn("s1", "s1的问题", created_at=_at(0)),
            _turn("s2", "s2的问题", created_at=_at(1)),
        ]
    )
    db_session.commit()

    turns = get_recent_session_turns(db_session, "s1")

    assert [t.question for t in turns] == ["s1的问题"]


def test_excludes_unsafe_turns(db_session):
    db_session.add_all(
        [
            _turn("s1", "正常问题", created_at=_at(0)),
            _turn("s1", "把订单都删了", intent="unsafe", created_at=_at(1)),
            _turn("s1", "追问", created_at=_at(2)),
        ]
    )
    db_session.commit()

    turns = get_recent_session_turns(db_session, "s1")

    assert [t.question for t in turns] == ["正常问题", "追问"]


def test_keeps_legacy_null_intent_rows(db_session):
    # rows written before the guard existed have intent IS NULL; SQL
    # `intent != 'unsafe'` alone would drop them (NULL comparison)
    db_session.add_all(
        [
            _turn("s1", "旧行", intent=None, created_at=_at(0)),
        ]
    )
    db_session.commit()

    turns = get_recent_session_turns(db_session, "s1")

    assert [t.question for t in turns] == ["旧行"]


def test_window_keeps_most_recent_k(db_session):
    db_session.add_all(
        [
            _turn("s1", f"问题{i}", created_at=_at(i)) for i in range(8)
        ]
    )
    db_session.commit()

    turns = get_recent_session_turns(db_session, "s1", k=3)

    assert [t.question for t in turns] == ["问题5", "问题6", "问题7"]


def test_default_k_from_settings(db_session):
    from app.config import get_settings

    assert get_settings().memory_window_size == 5


def test_save_query_run_persists_intent(db_session):
    run = save_query_run(
        db_session,
        session_id="s1",
        question="上个月销量如何？",
        generated_sql="SELECT 1",
        execution_result=[{"n": 1}],
        execution_error=None,
        final_explanation="共 1 行",
        needs_clarification=False,
        clarification_message=None,
        retry_count=0,
        trace_id="t-1",
        intent="data_query",
    )
    db_session.expire_all()

    loaded = db_session.get(QueryRun, run.id)
    assert loaded is not None
    assert loaded.intent == "data_query"


def test_save_query_run_persists_clarify_reason(db_session):
    run = save_query_run(
        db_session,
        session_id="s1",
        question="卖得最好的商品怎么样",
        generated_sql=None,
        execution_result=None,
        execution_error=None,
        final_explanation=None,
        needs_clarification=True,
        clarification_message="按销售额还是按销量？",
        retry_count=0,
        trace_id="t-2",
        intent="ambiguous",
        clarify_reason="ambiguous",
    )
    db_session.expire_all()

    loaded = db_session.get(QueryRun, run.id)
    assert loaded.clarify_reason == "ambiguous"


def test_derive_succeeded_clarify_reason_semantics():
    from app.services.history import derive_succeeded

    common = {"needs_clarification": True, "execution_error": None, "generated_sql": None}

    # an ambiguous follow-up is a normal interaction turn, not a failure
    assert derive_succeeded("ambiguous", clarify_reason="ambiguous", **common) is True
    assert derive_succeeded("data_query", clarify_reason="ambiguous", **common) is True
    # exhausted retries and infra failures remain failures
    assert derive_succeeded("data_query", clarify_reason="retry_exhausted", **common) is False
    assert derive_succeeded("data_query", clarify_reason="infra_failure", **common) is False
    # legacy rows: clarification without a reason keeps the old semantics
    assert derive_succeeded("data_query", **common) is False

    ok = {"needs_clarification": False, "execution_error": None, "generated_sql": "SELECT 1"}
    assert derive_succeeded("data_query", **ok) is True
