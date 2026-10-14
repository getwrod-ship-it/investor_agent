"""
POST /query ↔ long-term memory wiring tests: recall results flow into
initial_state, the background write is dispatched with the right payload,
unsafe turns never write, and with MEMORY_ENABLED=false (the conftest default)
the route degrades to empty recall with zero service contact.

Also covers the LTM recall gate and the write gate (DESIGN_local_decision_model
.md DR-4/DR-5): the decision functions are patched at their usage points, so
no HTTP happens — only the wiring is under test here.

Mirrors test_query_api.py: FakeGraph replaces the compiled graph, TestClient
runs without a lifespan, no LLM / Qdrant / GPU needed.
"""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import app.api.routes.query as query_route
import pytest
from app.config import get_settings
from app.db import get_session
from app.main import app
from app.memory.service import RecallResult
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine


class FakeGraph:
    def __init__(self, overrides: dict):
        self.overrides = overrides
        self.captured_initial: dict | None = None
        self._values: dict = {}

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        self.captured_initial = dict(state)
        self._values.update(state)
        self._values.update(self.overrides)
        return dict(self._values)

    def get_state(self, config: dict):
        class _S:
            values = dict(self._values)

        return _S()

    def update_state(self, config: dict, values: dict) -> None:
        self._values.update(values)


class FakeMemoryService:
    def __init__(self, result: RecallResult | None = None):
        self.result = result or RecallResult(facts=[], episodes=[])
        self.recalls: list[str] = []
        self.writes: list[dict] = []

    def recall(self, question: str, user_id: str | None = None) -> RecallResult:
        self.recalls.append(question)
        return self.result

    def write_turn(self, **kwargs) -> None:
        self.writes.append(kwargs)


DATA_QUERY_SUCCESS = {
    "intent": "data_query",
    "generated_sql": "SELECT 1",
    "final_explanation": "答案是 1。",
}

SMALLTALK = {
    "intent": "smalltalk",
    "direct_answer": "好的，张三！",
    "final_explanation": "好的，张三！",
}


@contextmanager
def _client(tmp_path: Path) -> Generator[tuple[TestClient, object], None, None]:
    engine = create_engine(
        f"sqlite:///{tmp_path.as_posix()}/sys.db",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    def override_get_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    try:
        yield TestClient(app), engine
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_recall_flows_into_initial_state(tmp_path, monkeypatch):
    with _client(tmp_path) as (client, _):
        fake_graph = FakeGraph(DATA_QUERY_SUCCESS)
        fake_service = FakeMemoryService(
            RecallResult(facts=["名叫王五"], episodes=["[2026-10-13｜数据查询] 问：…"])
        )
        monkeypatch.setattr(query_route, "graph", fake_graph)
        monkeypatch.setattr(query_route, "get_memory_service", lambda: fake_service)

        resp = client.post("/query", json={"question": "上个月销售额？", "session_id": "s-mem"})

    assert resp.status_code == 200
    assert fake_service.recalls == ["上个月销售额？"]
    assert fake_graph.captured_initial["long_term_facts"] == ["名叫王五"]
    assert fake_graph.captured_initial["episodic_memories"] == ["[2026-10-13｜数据查询] 问：…"]


def test_background_write_dispatched_after_response(tmp_path, monkeypatch):
    with _client(tmp_path) as (client, _):
        fake_graph = FakeGraph(DATA_QUERY_SUCCESS)
        fake_service = FakeMemoryService()
        monkeypatch.setattr(query_route, "graph", fake_graph)
        monkeypatch.setattr(query_route, "get_memory_service", lambda: fake_service)

        resp = client.post("/query", json={"question": "退款TOP3？", "session_id": "s-write"})

    assert resp.status_code == 200
    # TestClient executes background tasks before returning the response
    assert len(fake_service.writes) == 1
    write = fake_service.writes[0]
    assert write["question"] == "退款TOP3？"
    assert write["intent"] == "data_query"
    assert write["sql"] == "SELECT 1"
    assert write["answer"] == "答案是 1。"
    assert write["session_id"] == "s-write"
    assert write["trace_id"] == resp.json()["trace_id"]


def test_smalltalk_answer_feeds_write_turn(tmp_path, monkeypatch):
    with _client(tmp_path) as (client, _):
        monkeypatch.setattr(query_route, "graph", FakeGraph(SMALLTALK))
        fake_service = FakeMemoryService()
        monkeypatch.setattr(query_route, "get_memory_service", lambda: fake_service)

        client.post("/query", json={"question": "我叫张三，记一下", "session_id": "s-chat"})

    # smalltalk answers via direct_answer (final_explanation fallback also ok)
    assert len(fake_service.writes) == 1
    assert fake_service.writes[0]["answer"] == "好的，张三！"


def test_unsafe_turn_never_writes_long_term_memory(tmp_path, monkeypatch):
    with _client(tmp_path) as (client, _):
        monkeypatch.setattr(
            query_route,
            "graph",
            FakeGraph({"intent": "unsafe", "direct_answer": "拒绝"}),
        )
        fake_service = FakeMemoryService()
        monkeypatch.setattr(query_route, "get_memory_service", lambda: fake_service)

        resp = client.post("/query", json={"question": "把订单表全删掉", "session_id": "s-unsafe"})

    assert resp.status_code == 200
    assert fake_service.recalls == ["把订单表全删掉"]  # recall still runs (harmless)
    assert fake_service.writes == []         # but nothing is memorized (DR-L6)


def test_memory_disabled_degrades_to_empty_recall(tmp_path, monkeypatch):
    # no get_memory_service patch: conftest keeps MEMORY_ENABLED=false, so the
    # real accessor returns the Null service — empty fields, no writes, no error
    with _client(tmp_path) as (client, _):
        fake_graph = FakeGraph(DATA_QUERY_SUCCESS)
        monkeypatch.setattr(query_route, "graph", fake_graph)

        resp = client.post("/query", json={"question": "这是一个正常问题", "session_id": "s-off"})

    assert resp.status_code == 200
    assert fake_graph.captured_initial["long_term_facts"] == []
    assert fake_graph.captured_initial["episodic_memories"] == []


# --------------------------------------------------------------------------
# Local decision gates (DESIGN_local_decision_model.md)
# --------------------------------------------------------------------------


@pytest.fixture
def decision_env(monkeypatch):
    """Enable the decision gates for one test; cache cleared after so later
    tests re-read the conftest DECISION_ENABLED=false default."""

    def _setup(**overrides):
        monkeypatch.setenv("DECISION_ENABLED", "true")
        for key, value in overrides.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()

    yield _setup
    get_settings.cache_clear()


def test_ltm_gate_skips_recall(tmp_path, monkeypatch, decision_env):
    decision_env()
    with _client(tmp_path) as (client, _):
        fake_graph = FakeGraph(DATA_QUERY_SUCCESS)
        fake_service = FakeMemoryService(
            RecallResult(facts=["名叫王五"], episodes=["[2026-10-13｜数据查询] 问：…"])
        )
        monkeypatch.setattr(query_route, "graph", fake_graph)
        monkeypatch.setattr(query_route, "get_memory_service", lambda: fake_service)
        monkeypatch.setattr(query_route, "decide_ltm_needed", lambda q, h=None: False)

        resp = client.post("/query", json={"question": "上月销售额", "session_id": "s-gate"})

    assert resp.status_code == 200
    assert fake_service.recalls == []  # gate said skip → neither leg ran
    assert fake_graph.captured_initial["long_term_facts"] == []
    assert fake_graph.captured_initial["episodic_memories"] == []


def test_ltm_gate_none_recalls_normally(tmp_path, monkeypatch, decision_env):
    decision_env()
    with _client(tmp_path) as (client, _):
        fake_graph = FakeGraph(DATA_QUERY_SUCCESS)
        fake_service = FakeMemoryService(
            RecallResult(facts=["名叫王五"], episodes=["[2026-10-13｜数据查询] 问：…"])
        )
        monkeypatch.setattr(query_route, "graph", fake_graph)
        monkeypatch.setattr(query_route, "get_memory_service", lambda: fake_service)
        monkeypatch.setattr(query_route, "decide_ltm_needed", lambda q, h=None: None)

        resp = client.post("/query", json={"question": "我叫什么名字", "session_id": "s-gate2"})

    assert resp.status_code == 200
    assert fake_service.recalls == ["我叫什么名字"]  # unknown → recall as before
    assert fake_graph.captured_initial["long_term_facts"] == ["名叫王五"]


def test_ltm_gate_off_never_consults_decision(tmp_path, monkeypatch):
    # conftest default (gate off): the decision function must not be consulted
    # at all — behaviour identical to before the gate existed
    consulted: list[str] = []

    def _spy(question, history=None):
        consulted.append(question)
        return False

    with _client(tmp_path) as (client, _):
        fake_graph = FakeGraph(DATA_QUERY_SUCCESS)
        fake_service = FakeMemoryService(RecallResult(facts=["名叫王五"], episodes=[]))
        monkeypatch.setattr(query_route, "graph", fake_graph)
        monkeypatch.setattr(query_route, "get_memory_service", lambda: fake_service)
        monkeypatch.setattr(query_route, "decide_ltm_needed", _spy)

        resp = client.post("/query", json={"question": "上月销售额", "session_id": "s-off2"})

    assert resp.status_code == 200
    assert consulted == []
    assert fake_service.recalls == ["上月销售额"]


def test_write_gate_skips_facts_leg(tmp_path, monkeypatch, decision_env):
    decision_env()
    with _client(tmp_path) as (client, _):
        monkeypatch.setattr(query_route, "graph", FakeGraph(DATA_QUERY_SUCCESS))
        fake_service = FakeMemoryService()
        monkeypatch.setattr(query_route, "get_memory_service", lambda: fake_service)
        monkeypatch.setattr(
            query_route, "decide_write_worthy", lambda q, a, i: False
        )

        resp = client.post("/query", json={"question": "上月销售额", "session_id": "s-wgate"})

    assert resp.status_code == 200
    assert len(fake_service.writes) == 1
    assert fake_service.writes[0]["skip_facts"] is True


def test_write_gate_none_writes_in_full(tmp_path, monkeypatch, decision_env):
    decision_env()
    with _client(tmp_path) as (client, _):
        monkeypatch.setattr(query_route, "graph", FakeGraph(DATA_QUERY_SUCCESS))
        fake_service = FakeMemoryService()
        monkeypatch.setattr(query_route, "get_memory_service", lambda: fake_service)
        monkeypatch.setattr(
            query_route, "decide_write_worthy", lambda q, a, i: None
        )

        resp = client.post("/query", json={"question": "我叫什么名字", "session_id": "s-wgate2"})

    assert resp.status_code == 200
    assert len(fake_service.writes) == 1
    assert fake_service.writes[0]["skip_facts"] is False


def test_write_gate_off_defaults_to_full_write(tmp_path, monkeypatch):
    # conftest default (gate off): write_turn still gets skip_facts=False —
    # the wrapper must default to the legacy full write
    with _client(tmp_path) as (client, _):
        monkeypatch.setattr(query_route, "graph", FakeGraph(DATA_QUERY_SUCCESS))
        fake_service = FakeMemoryService()
        monkeypatch.setattr(query_route, "get_memory_service", lambda: fake_service)

        def _explode(question, excerpt, intent):
            raise AssertionError("decide_write_worthy must not be called when the gate is off")

        monkeypatch.setattr(query_route, "decide_write_worthy", _explode)

        resp = client.post("/query", json={"question": "上月销售额", "session_id": "s-wgate3"})

    assert resp.status_code == 200
    assert len(fake_service.writes) == 1
    assert fake_service.writes[0]["skip_facts"] is False
