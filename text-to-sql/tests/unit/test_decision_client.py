"""
Local decision client (app/agent/decision.py): request shaping, answer
parsing, thresholds, and the fail-open contract — every failure mode returns
None so the caller falls back to the legacy path. httpx is patched at the
usage point (same convention as the guard/rewrite get_structured_llm patches).
"""

import pytest
from app.agent.decision import (
    IntentDecision,
    decide_intent,
    decide_ltm_needed,
    decide_rewrite_needed,
    decide_write_worthy,
)
from app.agent.prompts.decision import INTENT_CRITERIA, build_state
from app.config import get_settings


class FakeResponse:
    def __init__(self, payload=None, status_error: Exception | None = None):
        self._payload = payload or {}
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    def json(self):
        return self._payload


class FakeClient:
    """Records the request; replies with a canned response or raises."""

    def __init__(self, response=None, error: Exception | None = None, timeout=None):
        self.response = response
        self.error = error
        self.requests: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None):
        self.requests.append({"url": url, "json": json})
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def decision_env(monkeypatch):
    """Enable the decision service for one test; cache cleared after so the
    conftest DECISION_ENABLED=false default is re-read by later tests."""

    def _setup(**overrides):
        monkeypatch.setenv("DECISION_ENABLED", "true")
        for key, value in overrides.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()

    yield _setup
    get_settings.cache_clear()


def _intent_response(**answer) -> FakeResponse:
    payload = {"answers": {"intent": answer or {}}}
    return FakeResponse(payload)


def test_disabled_makes_no_http_call(monkeypatch):
    # conftest default: DECISION_ENABLED=false → None, and the empty request
    # log proves no HTTP was attempted (the fake would have recorded it)
    client = FakeClient(
        FakeResponse({"answers": {"intent": {"type": "choice", "choice": "data_query", "confidence": 1.0}}})
    )
    monkeypatch.setattr("app.agent.decision.httpx.Client", lambda *a, **k: client)

    assert decide_intent("上月销售额") is None
    assert decide_ltm_needed("我叫什么") is None
    assert decide_write_worthy("问", "答", "data_query") is None
    assert decide_rewrite_needed("那前个月呢", []) is None
    assert client.requests == []


def test_guard_subflag_off_returns_none(monkeypatch, decision_env):
    decision_env(DECISION_GUARD_ENABLED="false")
    client = FakeClient(
        FakeResponse({"answers": {"intent": {"type": "choice", "choice": "data_query", "confidence": 1.0}}})
    )
    monkeypatch.setattr("app.agent.decision.httpx.Client", lambda *a, **k: client)

    assert decide_intent("上月销售额") is None
    assert client.requests == []


def test_intent_parses_choice_answer(monkeypatch, decision_env):
    decision_env()
    client = FakeClient(
        _intent_response(
            type="choice",
            choice="data_query",
            confidence=0.84,
            probabilities={"data_query": 0.84, "smalltalk": 0.06, "unsafe": 0.02, "ambiguous": 0.08},
        )
    )
    monkeypatch.setattr("app.agent.decision.httpx.Client", lambda *a, **k: client)

    result = decide_intent("上月销售额最高的三个商品", [], None)

    assert result == IntentDecision(
        intent="data_query",
        confidence=0.84,
        probabilities={"data_query": 0.84, "smalltalk": 0.06, "unsafe": 0.02, "ambiguous": 0.08},
    )
    # protocol shape: /v1/systemone with request-level state + one choice question
    req = client.requests[0]
    assert req["url"].endswith("/v1/systemone")
    assert req["json"]["state"]["user_query"] == "上月销售额最高的三个商品"
    question = req["json"]["questions"]["intent"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(INTENT_CRITERIA)


def test_intent_passes_history_into_state(monkeypatch, decision_env):
    decision_env()
    client = FakeClient(_intent_response(type="choice", choice="data_query", confidence=0.9, probabilities={}))
    monkeypatch.setattr("app.agent.decision.httpx.Client", lambda *a, **k: client)

    decide_intent(
        "那前个月呢",
        [{"question": "上月销售额", "sql": "SELECT 1", "answer": "42"}],
        "更早的会话摘要",
    )

    state = client.requests[0]["json"]["state"]
    assert state["recent_turns"] == [{"q": "上月销售额", "a": "42"}]
    assert state["history_summary"] == "更早的会话摘要"


@pytest.mark.parametrize(
    "answer",
    [
        {},  # missing everything
        {"type": "choice", "choice": "weird_label", "confidence": 0.9},  # unknown label
        {"type": "choice", "choice": "data_query"},  # missing confidence
        {"type": "choice", "choice": "data_query", "confidence": True},  # bool, not number
    ],
)
def test_intent_malformed_answer_returns_none(monkeypatch, decision_env, answer):
    decision_env()
    monkeypatch.setattr(
        "app.agent.decision.httpx.Client",
        lambda *a, **k: FakeClient(_intent_response(**answer)),
    )
    assert decide_intent("上月销售额") is None


@pytest.mark.parametrize("p,expected", [(0.59, False), (0.60, True), (0.99, True), (0.0, False)])
def test_ltm_gate_threshold_boundary(monkeypatch, decision_env, p, expected):
    decision_env()
    monkeypatch.setattr(
        "app.agent.decision.httpx.Client",
        lambda *a, **k: FakeClient(FakeResponse({"answers": {"ltm_need": {"type": "noul", "noul": p}}})),
    )
    assert decide_ltm_needed("上月销售额前十的商品") is expected


def test_noul_malformed_returns_none(monkeypatch, decision_env):
    decision_env()
    monkeypatch.setattr(
        "app.agent.decision.httpx.Client",
        lambda *a, **k: FakeClient(FakeResponse({"answers": {"ltm_need": {"type": "noul"}}})),
    )
    assert decide_ltm_needed("我叫什么") is None


@pytest.mark.parametrize("p,expected", [(0.91, True), (0.5, True), (0.49, False)])
def test_write_worthy_plain_cut(monkeypatch, decision_env, p, expected):
    decision_env()
    monkeypatch.setattr(
        "app.agent.decision.httpx.Client",
        lambda *a, **k: FakeClient(FakeResponse({"answers": {"write_worthy": {"type": "noul", "noul": p}}})),
    )
    assert decide_write_worthy("以后叫我小张", "好的，小张", "smalltalk") is expected


@pytest.mark.parametrize("p,expected", [(0.8, True), (0.5, True), (0.3, False)])
def test_rewrite_needed_plain_cut(monkeypatch, decision_env, p, expected):
    decision_env()
    monkeypatch.setattr(
        "app.agent.decision.httpx.Client",
        lambda *a, **k: FakeClient(FakeResponse({"answers": {"rewrite_need": {"type": "noul", "noul": p}}})),
    )
    assert decide_rewrite_needed("那前个月呢", [{"question": "上月销量", "sql": None, "answer": "42"}]) is expected


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("decision service timed out"),
        ConnectionError("connection refused"),
        Exception("httpx internal"),
    ],
)
def test_transport_errors_return_none(monkeypatch, decision_env, error):
    decision_env()
    monkeypatch.setattr(
        "app.agent.decision.httpx.Client", lambda *a, **k: FakeClient(error=error)
    )
    assert decide_intent("上月销售额") is None
    assert decide_ltm_needed("我叫什么") is None


def test_http_500_returns_none(monkeypatch, decision_env):
    decision_env()
    monkeypatch.setattr(
        "app.agent.decision.httpx.Client",
        lambda *a, **k: FakeClient(FakeResponse(status_error=RuntimeError("500"))),
    )
    assert decide_intent("上月销售额") is None


def test_missing_answers_field_returns_none(monkeypatch, decision_env):
    decision_env()
    monkeypatch.setattr(
        "app.agent.decision.httpx.Client",
        lambda *a, **k: FakeClient(FakeResponse({"usage": {"input_tokens": 42}})),
    )
    assert decide_intent("上月销售额") is None


def test_write_worthy_state_carries_excerpt_and_intent(monkeypatch, decision_env):
    decision_env()
    client = FakeClient(
        FakeResponse({"answers": {"write_worthy": {"type": "noul", "noul": 0.9}}})
    )
    monkeypatch.setattr("app.agent.decision.httpx.Client", lambda *a, **k: client)

    decide_write_worthy("问" * 10, "答" * 500, "data_query")

    state = client.requests[0]["json"]["state"]
    assert state["intent"] == "data_query"
    assert state["answer_excerpt"] == "答" * 300  # budget: 300 chars
    assert "recent_turns" in state


def test_build_state_budget():
    history = [
        {"question": f"问题{i}x" + "很" * 300, "sql": None, "answer": "答" * 300}
        for i in range(5)
    ]
    state = build_state("问" * 800, history, "摘要" + "长" * 500)
    # only the last 3 turns, each side truncated to 120 chars
    assert len(state["recent_turns"]) == 3
    assert state["recent_turns"][0]["q"] == ("问题2x" + "很" * 300)[:120]
    assert all(len(t["q"]) <= 120 and len(t["a"]) <= 120 for t in state["recent_turns"])
    assert state["user_query"] == "问" * 500
    assert state["history_summary"] == ("摘要" + "长" * 500)[:300]


def test_build_state_drops_empty_summary():
    state = build_state("q", [], None)
    assert "history_summary" not in state
    assert state["recent_turns"] == []
