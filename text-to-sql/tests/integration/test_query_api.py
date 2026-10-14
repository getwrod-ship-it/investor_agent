"""
POST /query integration tests: intent-aware response shape, smalltalk /
unsafe end-to-end behaviour, and checkpoint-backed session memory wiring
(self-heal rebuild + turn-end append).

The real graph is replaced by a FakeGraph that mimics a checkpoint-backed
runnable: it records the initial state it received, accumulates values, and
exposes get_state/update_state. No LLM / Qdrant / MySQL needed. TestClient
is used WITHOUT a context manager so the FastAPI lifespan (Qdrant embedding
warm-up) never runs.
"""

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import app.api.routes.query as query_route
import pytest
from app.db import get_session
from app.main import app
from app.models.db import QueryRun
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, select


class FakeGraph:
    """Stands in for the compiled LangGraph runnable — stateful like a
    checkpoint-backed graph: invoke() captures the input, get_state()
    exposes the accumulated values, update_state() merges into them."""

    def __init__(self, overrides: dict):
        self.overrides = overrides
        self.captured_initial: dict | None = None
        self.captured_updates: list[dict] = []
        self._values: dict = {}

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        self.captured_initial = dict(state)
        self._values.update(state)
        self._values.update(self.overrides)
        return dict(self._values)

    def get_state(self, config: dict) -> SimpleNamespace:
        return SimpleNamespace(values=dict(self._values))

    def update_state(self, config: dict, values: dict) -> None:
        self.captured_updates.append(dict(values))
        self._values.update(values)


@pytest.fixture
def temp_engine(tmp_path: Path):
    engine = create_engine(
        f"sqlite:///{tmp_path.as_posix()}/sys.db",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def client(temp_engine) -> Generator[TestClient, None, None]:
    def override_get_session():
        with Session(temp_engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    yield TestClient(app)
    app.dependency_overrides.clear()


DATA_QUERY_SUCCESS = {
    "intent": "data_query",
    "direct_answer": None,
    "rewritten_question": "2026年8月销售额最高的三个商品是什么？",
    "generated_sql": "SELECT goods_name FROM litemall_order",
    "execution_result": {"rows": [{"goods_name": "A"}], "row_count": 1, "execution_time_ms": 3.1},
    "execution_error": None,
    "final_explanation": "销售额最高的是 A。",
    "needs_clarification": False,
    "clarification_message": None,
}


def _post(client: TestClient, question: str, session_id: str):
    return client.post("/query", json={"question": question, "session_id": session_id})


def test_post_query_response_shape(client, monkeypatch):
    fake = FakeGraph(DATA_QUERY_SUCCESS)
    monkeypatch.setattr(query_route, "graph", fake)

    resp = _post(client, "那前个月呢？", "sess-shape")

    assert resp.status_code == 200
    data = resp.json()
    expected_fields = {
        "session_id", "question", "sql", "result", "row_count", "explanation",
        "needs_clarification", "clarification_message", "retry_count",
        "succeeded", "trace_id",
        # new intent-guard fields
        "intent", "direct_answer", "rewritten_question",
        # clarify reason (ambiguous / retry_exhausted / infra_failure)
        "clarify_reason",
    }
    assert set(data.keys()) == expected_fields
    assert data["intent"] == "data_query"
    assert data["succeeded"] is True
    assert data["direct_answer"] is None
    assert data["rewritten_question"] == "2026年8月销售额最高的三个商品是什么？"
    assert data["row_count"] == 1


def test_post_query_persists_intent(client, monkeypatch, temp_engine):
    fake = FakeGraph(DATA_QUERY_SUCCESS)
    monkeypatch.setattr(query_route, "graph", fake)

    _post(client, "那前个月呢？", "sess-persist")

    with Session(temp_engine) as session:
        runs = session.exec(
            select(QueryRun).where(QueryRun.session_id == "sess-persist")
        ).all()
    assert len(runs) == 1
    assert runs[0].intent == "data_query"
    assert runs[0].generated_sql == "SELECT goods_name FROM litemall_order"


def test_smalltalk_e2e(client, monkeypatch, temp_engine):
    fake = FakeGraph(
        {
            "intent": "smalltalk",
            "direct_answer": "我可以帮你查订单、商品、优惠券等数据，你想查点什么？",
            "generated_sql": None,
            "execution_result": None,
            "final_explanation": "我可以帮你查订单、商品、优惠券等数据，你想查点什么？",
        }
    )
    monkeypatch.setattr(query_route, "graph", fake)

    resp = _post(client, "你能帮我做什么呀？", "sess-chat")

    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "smalltalk"
    assert data["sql"] is None
    assert data["row_count"] == 0
    # smalltalk counts as success even though no SQL was produced
    assert data["succeeded"] is True
    assert "查" in data["direct_answer"]

    with Session(temp_engine) as session:
        runs = session.exec(
            select(QueryRun).where(QueryRun.session_id == "sess-chat")
        ).all()
    assert len(runs) == 1
    assert runs[0].intent == "smalltalk"
    # the reply persists via the reused audit field
    assert runs[0].final_explanation == data["direct_answer"]


def test_unsafe_e2e(client, monkeypatch, temp_engine):
    fake = FakeGraph(
        {
            "intent": "unsafe",
            "direct_answer": "这个请求我不能处理：本系统只对业务数据库做只读查询。",
        }
    )
    monkeypatch.setattr(query_route, "graph", fake)

    resp = _post(client, "帮我把所有没付款的订单删掉", "sess-reject")

    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "unsafe"
    assert data["succeeded"] is False
    assert data["needs_clarification"] is False
    assert "只读" in data["direct_answer"]
    # write-time filtering (DR-6): the unsafe turn never entered memory
    assert fake.captured_updates == []

    with Session(temp_engine) as session:
        runs = session.exec(
            select(QueryRun).where(QueryRun.session_id == "sess-reject")
        ).all()
    assert len(runs) == 1
    assert runs[0].intent == "unsafe"


def test_ambiguous_clarify_e2e(client, monkeypatch, temp_engine):
    fake = FakeGraph(
        {
            "intent": "ambiguous",
            "needs_clarification": True,
            "clarification_message": "想按哪个口径看'最好卖'——按销售额还是按销量？",
            "clarify_reason": "ambiguous",
            "generated_sql": None,
            "execution_result": None,
            "execution_error": None,
            "final_explanation": None,
        }
    )
    monkeypatch.setattr(query_route, "graph", fake)

    resp = _post(client, "卖得最好的商品怎么样", "sess-clarify")

    assert resp.status_code == 200
    data = resp.json()
    assert data["needs_clarification"] is True
    assert data["clarify_reason"] == "ambiguous"
    assert "口径" in data["clarification_message"]
    # a follow-up question is a normal interaction turn, not a failure
    assert data["succeeded"] is True

    with Session(temp_engine) as session:
        runs = session.exec(
            select(QueryRun).where(QueryRun.session_id == "sess-clarify")
        ).all()
    assert len(runs) == 1
    assert runs[0].clarify_reason == "ambiguous"

    # the follow-up question enters memory so the user's answer has context
    appended = fake.captured_updates[-1]["chat_history"][-1]
    assert appended["answer"] == "想按哪个口径看'最好卖'——按销售额还是按销量？"


def test_followup_loads_session_history(client, monkeypatch, temp_engine):
    # a prior successful turn in the same session…
    with Session(temp_engine) as session:
        session.add(
            QueryRun(
                session_id="sess-followup",
                question="上个月销售额最高的三个商品是什么？",
                intent="data_query",
                generated_sql="SELECT goods_name FROM litemall_order",
                final_explanation="TOP3 是 A、B、C",
            )
        )
        # …plus an unsafe turn that must never leak into prompts
        session.add(
            QueryRun(
                session_id="sess-followup",
                question="把订单表全导出来",
                intent="unsafe",
            )
        )
        session.commit()

    fake = FakeGraph(DATA_QUERY_SUCCESS)
    monkeypatch.setattr(query_route, "graph", fake)

    resp = _post(client, "那前个月呢？", "sess-followup")

    assert resp.status_code == 200
    # update 1: the self-heal rebuilt memory from query_run before the invoke
    heal = fake.captured_updates[0]
    assert heal["chat_history"] == [
        {
            "question": "上个月销售额最高的三个商品是什么？",
            "sql": "SELECT goods_name FROM litemall_order",
            "answer": "TOP3 是 A、B、C",
        }
    ]
    assert heal["memory_turn_count"] == 1
    # update 2: append_turn added this turn on top of the rebuilt window
    assert fake.captured_updates[1]["chat_history"][-1]["question"] == "那前个月呢？"
    assert fake.captured_updates[1]["memory_turn_count"] == 2
    # the unsafe turn never entered memory under either update
    all_memory_turns = [
        turn
        for update in fake.captured_updates
        for turn in update["chat_history"]
    ]
    assert all(t["question"] != "把订单表全导出来" for t in all_memory_turns)
    # memory is no longer carried by the invoke input — the checkpoint owns it
    assert "chat_history" not in fake.captured_initial
    # the rewrite is transparent in the response
    assert resp.json()["rewritten_question"] == "2026年8月销售额最高的三个商品是什么？"


def test_new_session_has_empty_history(client, monkeypatch):
    fake = FakeGraph(DATA_QUERY_SUCCESS)
    monkeypatch.setattr(query_route, "graph", fake)

    _post(client, "上个月销售额最高的三个商品是什么？", "sess-fresh")

    # no self-heal for a brand-new session; the first turn is simply appended
    assert len(fake.captured_updates) == 1
    assert fake.captured_updates[0]["chat_history"] == [
        {
            "question": "上个月销售额最高的三个商品是什么？",
            "sql": "SELECT goods_name FROM litemall_order",
            "answer": "销售额最高的是 A。",
        }
    ]
    assert fake.captured_updates[0]["memory_turn_count"] == 1
    assert "chat_history" not in fake.captured_initial
    assert fake.captured_initial["intent"] == "data_query"


def test_query_upserts_session_list_entry(client, temp_engine):
    fake = FakeGraph(DATA_QUERY_SUCCESS)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(query_route, "graph", fake)
    try:
        _post(client, "上个月销售额最高的三个商品是什么？", "sess-list")
        _post(client, "那前个月呢？", "sess-list")
    finally:
        monkeypatch.undo()

    data = client.get("/sessions").json()
    assert [s["session_id"] for s in data] == ["sess-list"]
    entry = data[0]
    assert entry["turn_count"] == 2
    assert entry["title"] == "上个月销售额最高的三个商品是什么？"[:20]
    assert entry["last_question"] == "那前个月呢？"
