"""
Conversation memory service: turn-end append (window trim, unsafe skip,
rolling summary fail-open) and the ensure_thread_state self-heal.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import app.services.conversation as conversation
from app.config import get_settings
from app.models.db import QueryRun
from app.services.conversation import (
    append_turn,
    ensure_thread_state,
    update_summary,
)

_BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


class MemoryGraph:
    """Minimal checkpoint-backed graph double."""

    def __init__(self, values: dict | None = None):
        self.values = dict(values or {})
        self.updates: list[dict] = []

    def get_state(self, config: dict) -> SimpleNamespace:
        return SimpleNamespace(values=dict(self.values))

    def update_state(self, config: dict, values: dict) -> None:
        self.updates.append(dict(values))
        self.values.update(values)


def _turn(question: str, sql: str | None = None) -> dict:
    return {"question": question, "sql": sql, "answer": "答"}


def _final_state(**overrides) -> dict:
    base = {
        "question": "这个月有多少订单？",
        "intent": "data_query",
        "generated_sql": "SELECT COUNT(*) FROM litemall_order",
        "final_explanation": "共 123 条。",
    }
    base.update(overrides)
    return base


def _seed_runs(db_session, session_id: str, count: int, *, include_unsafe=False):
    rows = [
        QueryRun(
            session_id=session_id,
            question=f"问题{i}",
            intent="data_query",
            generated_sql=f"SELECT {i}",
            final_explanation=f"答案{i}",
            created_at=_BASE_TIME + timedelta(minutes=i),
        )
        for i in range(1, count + 1)
    ]
    if include_unsafe:
        rows.append(
            QueryRun(
                session_id=session_id,
                question="危险操作",
                intent="unsafe",
                created_at=_BASE_TIME + timedelta(minutes=count + 1),
            )
        )
    for row in rows:
        db_session.add(row)
    db_session.commit()


# ---------------------------------------------------------------------------
# append_turn
# ---------------------------------------------------------------------------


def test_append_turn_appends_and_bumps_watermark():
    graph = MemoryGraph(
        {"chat_history": [], "history_summary": None, "memory_turn_count": 0}
    )
    append_turn(graph, {"configurable": {"thread_id": "s"}}, _final_state())

    update = graph.updates[0]
    assert update["chat_history"] == [
        {
            "question": "这个月有多少订单？",
            "sql": "SELECT COUNT(*) FROM litemall_order",
            "answer": "共 123 条。",
        }
    ]
    assert update["memory_turn_count"] == 1


def test_append_turn_skips_unsafe_entirely():
    graph = MemoryGraph({"chat_history": [], "memory_turn_count": 3})
    append_turn(
        graph, {"configurable": {"thread_id": "s"}}, _final_state(intent="unsafe")
    )
    assert graph.updates == []


def test_append_turn_uses_clarification_message_for_clarify_turns():
    graph = MemoryGraph()
    append_turn(
        graph,
        {"configurable": {"thread_id": "s"}},
        _final_state(
            intent="ambiguous",
            generated_sql=None,
            final_explanation=None,
            clarification_message="想按哪个口径看'最好卖'——按销售额还是按销量？",
        ),
    )

    turn = graph.updates[0]["chat_history"][0]
    # without the clarification message the follow-up answer would arrive
    # with no memory of what was asked
    assert turn["answer"] == "想按哪个口径看'最好卖'——按销售额还是按销量？"
    assert turn["sql"] is None


def test_append_turn_truncates_answer_to_200_chars():
    graph = MemoryGraph()
    append_turn(graph, {"configurable": {"thread_id": "s"}},
                _final_state(final_explanation="长" * 500))
    assert len(graph.updates[0]["chat_history"][0]["answer"]) == 200


def test_append_turn_trims_window_and_folds_summary(monkeypatch):
    history = [_turn(f"问题{i}", sql=f"SELECT {i}") for i in range(1, 6)]
    graph = MemoryGraph(
        {"chat_history": history, "history_summary": "旧摘要", "memory_turn_count": 5}
    )
    captured: dict = {}

    def fake_update_summary(old, fallen):
        captured["old"] = old
        captured["fallen"] = fallen
        return "新摘要"

    monkeypatch.setattr(conversation, "update_summary", fake_update_summary)

    append_turn(graph, {"configurable": {"thread_id": "s"}}, _final_state())

    update = graph.updates[0]
    assert len(update["chat_history"]) == 5
    assert update["chat_history"][0]["question"] == "问题2"  # 问题1 fell out
    assert update["chat_history"][-1]["question"] == "这个月有多少订单？"
    assert captured["old"] == "旧摘要"
    assert [t["question"] for t in captured["fallen"]] == ["问题1"]
    assert update["history_summary"] == "新摘要"
    assert update["memory_turn_count"] == 6


def test_append_turn_without_summary_drops_fallen_turn(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "memory_summary_enabled", False)

    history = [_turn(f"问题{i}", sql=f"SELECT {i}") for i in range(1, 6)]
    graph = MemoryGraph(
        {"chat_history": history, "history_summary": None, "memory_turn_count": 5}
    )
    append_turn(graph, {"configurable": {"thread_id": "s"}}, _final_state())

    update = graph.updates[0]
    assert len(update["chat_history"]) == 5
    assert update["history_summary"] is None  # compression disabled
    assert update["memory_turn_count"] == 6


# ---------------------------------------------------------------------------
# update_summary
# ---------------------------------------------------------------------------


def test_update_summary_fail_open_keeps_old_summary(monkeypatch):
    class BoomLLM:
        def invoke(self, prompt):
            raise RuntimeError("api down")

    monkeypatch.setattr(conversation, "get_chat_llm", lambda temperature=0: BoomLLM())
    assert update_summary("旧摘要", [_turn("问题1")]) == "旧摘要"


def test_update_summary_compresses_and_truncates(monkeypatch):
    captured: dict = {}

    class FakeLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return SimpleNamespace(content="摘要正文" * 500)

    monkeypatch.setattr(conversation, "get_chat_llm", lambda temperature=0: FakeLLM())

    result = update_summary(
        None, [{"question": "问题1", "sql": "SELECT 1", "answer": "答案1"}]
    )

    assert "问题1" in captured["prompt"]  # the fallen turn reaches the prompt
    max_chars = get_settings().memory_summary_max_chars
    assert result == ("摘要正文" * 500)[:max_chars]


# ---------------------------------------------------------------------------
# ensure_thread_state
# ---------------------------------------------------------------------------


def test_ensure_noop_for_brand_new_session(db_session):
    graph = MemoryGraph()
    ensure_thread_state(db_session, graph, {"configurable": {"thread_id": "s"}}, "s")
    assert graph.updates == []


def test_ensure_noop_when_watermark_consistent(db_session):
    _seed_runs(db_session, "s", 1)
    graph = MemoryGraph(
        {"memory_turn_count": 1, "chat_history": [_turn("问题1")], "history_summary": None}
    )
    ensure_thread_state(db_session, graph, {"configurable": {"thread_id": "s"}}, "s")
    assert graph.updates == []


def test_ensure_rebuilds_window_from_query_run(db_session):
    _seed_runs(db_session, "s", 2, include_unsafe=True)
    graph = MemoryGraph()  # empty checkpoint → watermark 0 (legacy session)

    ensure_thread_state(db_session, graph, {"configurable": {"thread_id": "s"}}, "s")

    update = graph.updates[0]
    assert [t["question"] for t in update["chat_history"]] == ["问题1", "问题2"]
    assert update["memory_turn_count"] == 2  # unsafe turn never counted


def test_ensure_rebuild_includes_clarify_message(db_session):
    db_session.add(
        QueryRun(
            session_id="s",
            question="卖得最好的商品怎么样",
            intent="ambiguous",
            needs_clarification=True,
            clarification_message="按销售额还是按销量？",
            created_at=_BASE_TIME,
        )
    )
    db_session.commit()
    graph = MemoryGraph()

    ensure_thread_state(db_session, graph, {"configurable": {"thread_id": "s"}}, "s")

    update = graph.updates[0]
    assert update["chat_history"][0]["answer"] == "按销售额还是按销量？"
    assert update["memory_turn_count"] == 1


def test_ensure_summarizes_pre_window_turns(db_session, monkeypatch):
    _seed_runs(db_session, "s", 6)
    captured: dict = {}

    def fake_summarize_once(runs):
        captured["runs"] = runs
        return "重建摘要"

    monkeypatch.setattr(conversation, "summarize_once", fake_summarize_once)
    graph = MemoryGraph()

    ensure_thread_state(db_session, graph, {"configurable": {"thread_id": "s"}}, "s")

    update = graph.updates[0]
    assert len(update["chat_history"]) == 5  # window stays k
    assert update["memory_turn_count"] == 6
    assert update["history_summary"] == "重建摘要"
    assert len(captured["runs"]) == 1  # only the pre-window turn
    assert captured["runs"][0].question == "问题1"


def test_ensure_reuses_summary_when_watermark_stale_but_nonzero(db_session):
    # scenario: crash after update_state but before save_query_run — the
    # summary already covers the fallen turns, so it must be reused
    _seed_runs(db_session, "s", 6)
    graph = MemoryGraph(
        {"memory_turn_count": 5, "chat_history": [_turn("旧")] * 5, "history_summary": "旧摘要"}
    )

    ensure_thread_state(db_session, graph, {"configurable": {"thread_id": "s"}}, "s")

    update = graph.updates[0]
    assert update["history_summary"] == "旧摘要"  # reused, not re-summarised
    assert update["memory_turn_count"] == 6
    assert update["chat_history"][-1]["question"] == "问题6"
