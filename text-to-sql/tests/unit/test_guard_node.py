"""
GUARD node: LLM classification with fail-open behaviour, plus the local
decision cascade (high-confidence adoption / low-confidence fallback / shadow).
"""

import logging

import pytest
from app.agent.decision import IntentDecision
from app.agent.nodes.guard import IntentClassification, guard
from app.config import get_settings
from langchain_core.messages import AIMessage


class FakeStructuredLLM:
    """Mimics get_structured_llm(...).invoke() → {"parsed": ..., "raw": ...}."""

    def __init__(self, parsed=None, error: Exception | None = None):
        self.parsed = parsed
        self.error = error
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return {"parsed": self.parsed, "raw": AIMessage(content="{}")}


def _patch_llm(monkeypatch, fake: FakeStructuredLLM):
    monkeypatch.setattr(
        "app.agent.nodes.guard.get_structured_llm", lambda schema: fake
    )


def _patch_decision(monkeypatch, decision):
    """Patch decide_intent at its usage point inside the guard node."""
    monkeypatch.setattr(
        "app.agent.nodes.guard.decide_intent", lambda question, history, summary=None: decision
    )


@pytest.fixture
def decision_env(monkeypatch):
    """Override decision settings for one test; cache cleared after so later
    tests re-read the conftest DECISION_ENABLED=false default."""

    def _setup(**overrides):
        monkeypatch.setenv("DECISION_ENABLED", "true")
        for key, value in overrides.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()

    yield _setup
    get_settings.cache_clear()


@pytest.mark.parametrize(
    "intent", ["data_query", "smalltalk", "unsafe", "ambiguous"]
)
def test_guard_sets_intent_from_llm(make_state, monkeypatch, intent):
    parsed = IntentClassification(
        intent=intent, reason="测试", is_followup=False
    )
    fake = FakeStructuredLLM(parsed=parsed)
    _patch_llm(monkeypatch, fake)

    result = guard(make_state(question="随便问点什么"))

    assert result["intent"] == intent
    assert len(fake.calls) == 1


def test_guard_prompt_contains_ambiguous_rules(make_state, monkeypatch):
    parsed = IntentClassification(intent="ambiguous", reason="口径二义", is_followup=False)
    fake = FakeStructuredLLM(parsed=parsed)
    _patch_llm(monkeypatch, fake)

    guard(make_state(question="卖得最好的商品怎么样"))

    system_content = fake.calls[0][0]["content"]
    assert "ambiguous" in system_content
    # high-threshold rule: prefer data_query over a false ambiguous
    assert "拿不准 ambiguous 还是 data_query 时，判 data_query" in system_content


def test_guard_prompt_contains_history(make_state, monkeypatch):
    parsed = IntentClassification(intent="data_query", reason="追问", is_followup=True)
    fake = FakeStructuredLLM(parsed=parsed)
    _patch_llm(monkeypatch, fake)

    history = [
        {
            "question": "上个月销售额最高的三个商品是什么？",
            "sql": "SELECT 1",
            "answer": "答案是A/B/C",
        }
    ]
    guard(make_state(chat_history=history))  # type: ignore[arg-type]

    messages = fake.calls[0]
    system_content = messages[0]["content"]
    assert "上个月销售额最高的三个商品是什么" in system_content
    assert "SELECT 1" in system_content


def test_guard_fail_open_on_llm_error(make_state, monkeypatch):
    _patch_llm(monkeypatch, FakeStructuredLLM(error=RuntimeError("api down")))

    result = guard(make_state(question="上个月销量如何？"))

    # Never block a real data question because the guard broke
    assert result["intent"] == "data_query"


def test_guard_fail_open_when_parsing_fails(make_state, monkeypatch):
    # json_mode can return parsed=None when the model output is not valid JSON
    _patch_llm(monkeypatch, FakeStructuredLLM(parsed=None))

    result = guard(make_state(question="上个月销量如何？"))

    assert result["intent"] == "data_query"


# --------------------------------------------------------------------------
# Local decision cascade (DESIGN_local_decision_model.md DR-3)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("intent", ["data_query", "smalltalk", "unsafe", "ambiguous"])
def test_local_high_confidence_decision_adopted_without_llm(
    make_state, monkeypatch, intent
):
    _patch_decision(monkeypatch, IntentDecision(intent, 0.9, {intent: 0.9}))

    def explode(schema):
        raise AssertionError(
            "get_structured_llm must not be called when the local decision is adopted"
        )

    monkeypatch.setattr("app.agent.nodes.guard.get_structured_llm", explode)

    result = guard(make_state(question="随便问点什么"))

    assert result["intent"] == intent


def test_local_confidence_at_threshold_is_adopted(make_state, monkeypatch):
    # adoption rule is confidence >= threshold (0.70 by default)
    _patch_decision(monkeypatch, IntentDecision("unsafe", 0.70, {"unsafe": 0.70}))

    def explode(schema):
        raise AssertionError("LLM must not be called when the local decision is adopted")

    monkeypatch.setattr("app.agent.nodes.guard.get_structured_llm", explode)

    result = guard(make_state(question="删库"))

    assert result["intent"] == "unsafe"


def test_local_low_confidence_falls_back_to_deepseek(make_state, monkeypatch):
    _patch_decision(monkeypatch, IntentDecision("smalltalk", 0.5, {"smalltalk": 0.5}))
    fake = FakeStructuredLLM(
        parsed=IntentClassification(intent="data_query", reason="低置信回退", is_followup=False)
    )
    _patch_llm(monkeypatch, fake)

    result = guard(make_state(question="那前个月呢？"))

    assert result["intent"] == "data_query"
    assert len(fake.calls) == 1


def test_local_decision_none_falls_back_to_deepseek(make_state, monkeypatch):
    # decision service down / disabled → None → pure legacy path
    _patch_decision(monkeypatch, None)
    fake = FakeStructuredLLM(
        parsed=IntentClassification(intent="smalltalk", reason="服务挂了", is_followup=False)
    )
    _patch_llm(monkeypatch, fake)

    result = guard(make_state(question="你是谁"))

    assert result["intent"] == "smalltalk"
    assert len(fake.calls) == 1


def test_shadow_mode_never_adopts_but_logs_comparison(
    make_state, monkeypatch, decision_env, caplog
):
    decision_env(DECISION_SHADOW_MODE="true")
    # a decision confident enough to be adopted — shadow must suppress adoption
    _patch_decision(monkeypatch, IntentDecision("data_query", 0.95, {"data_query": 0.95}))
    fake = FakeStructuredLLM(
        parsed=IntentClassification(intent="data_query", reason="影子对照", is_followup=False)
    )
    _patch_llm(monkeypatch, fake)

    with caplog.at_level(logging.INFO, logger="app.agent.nodes.guard"):
        result = guard(make_state(question="上月销售额"))

    # behaviour identical to the legacy path: DeepSeek answered
    assert result["intent"] == "data_query"
    assert len(fake.calls) == 1
    shadow_logs = [r.message for r in caplog.records if "GUARD[shadow]" in r.message]
    assert shadow_logs and "local=data_query(0.950) vs deepseek=data_query match=True" in shadow_logs[0]
