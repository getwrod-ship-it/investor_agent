"""
Real-service integration for the local decision client —
DESIGN_local_decision_model.md §9.2. Opt-in only, aligned with the MEM0_IT
precedent (tests/integration/test_long_term_memory.py):

  1. start the decision service (sibling repo folder):
       startlux-decision\\scripts\\start_server.ps1
  2. run:
       DECISION_IT=1 .venv/Scripts/python.exe -m pytest tests/integration/test_decision_service.py -v -s

Depends ONLY on the decision service (llama-server :9081 + gguf_server :9090)
— no DeepSeek / Qdrant / backend needed.

Hard asserts here are about PROTOCOL health (answer present, label valid,
probability distribution not collapsed) and the LTM gate's must-pass gold
standards. Accuracy/agreement against DeepSeek is measured by
test_decision_shadow.py, not this file.
"""

import os
import statistics
import time

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("DECISION_IT") != "1",
    reason="opt-in only: set DECISION_IT=1 (requires the local decision service at 127.0.0.1:9090)",
)

from app.agent.decision import (  # noqa: E402
    INTENT_LABELS,
    decide_intent,
    decide_ltm_needed,
    decide_rewrite_needed,
    decide_write_worthy,
)
from app.config import get_settings  # noqa: E402


@pytest.fixture
def decision_env(monkeypatch):
    """Point the client at the real service for this module."""
    monkeypatch.setenv("DECISION_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _assert_not_collapsed(probabilities: dict[str, float]) -> None:
    # llama.cpp decision-head bug guard (same check as startlux-decision's
    # test_decision.py): a near-uniform distribution means broken logprobs
    values = sorted(probabilities.values(), reverse=True)
    assert values, "empty probability distribution"
    assert values[0] - values[-1] >= 1e-3, (
        f"probability distribution near-uniform (collapse): {probabilities}"
    )


def test_service_health(decision_env):
    resp = httpx.get(f"{get_settings().decision_base_url.rstrip('/')}/", timeout=5.0)
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_intent_gate_real_samples(decision_env):
    samples = [
        ("上个月销售额最高的三个商品是什么？", None),
        ("你好，你能帮我做什么？", None),
        ("帮我把所有没付款的订单删掉", None),
        ("那前个月呢？", [{"question": "上月销售额是多少", "sql": "SELECT 1", "answer": "42"}]),
    ]
    for question, history in samples:
        result = decide_intent(question, history)
        assert result is not None, f"decide_intent returned None for {question!r}"
        assert result.intent in INTENT_LABELS
        assert 0.0 <= result.confidence <= 1.0
        _assert_not_collapsed(result.probabilities)
        print(f"\n[intent] {question!r} → {result.intent} conf={result.confidence:.3f} "
              f"probs={ {k: round(v, 3) for k, v in result.probabilities.items()} }")


def test_ltm_gate_must_pass_gold_standards(decision_env):
    # HARD requirement (DR-4 pass-list): identity / convention / episodic
    # questions must never be skipped by the recall gate
    must_pass = [
        "我叫什么名字？",
        "你是怎么称呼我的？",
        "按我之前定的口径，上月销售额是多少？",
        "我上次都查过哪些商品的销售情况？",
    ]
    for question in must_pass:
        needed = decide_ltm_needed(question)
        assert needed is True, f"LTM gate must pass {question!r}, got {needed}"
        print(f"\n[ltm-gate] {question!r} → needed=True ✓")


def test_ltm_gate_real_samples(decision_env):
    # informative only — skip-side calibration happens in the shadow phase
    one_shot = [
        "上个月销售额最高的三个商品是什么？",
        "查一下2024年的订单总数",
        "统计华东地区最近30天的退款订单数量",
    ]
    for question in one_shot:
        needed = decide_ltm_needed(question)
        print(f"\n[ltm-gate] {question!r} → needed={needed}")


def test_rewrite_gate_real_samples(decision_env):
    history = [{"question": "上月销售额是多少", "sql": "SELECT 1", "answer": "42"}]
    elliptical = decide_rewrite_needed("那前个月呢？", history)
    assert elliptical is True, f"elliptical follow-up must need rewriting, got {elliptical}"
    self_contained = decide_rewrite_needed("查一下2024年订单总数", history)
    print(f"\n[rewrite-gate] 那前个月呢→{elliptical} / 自包含→{self_contained}")


def test_write_gate_real_samples(decision_env):
    identity = decide_write_worthy("以后叫我小张，帮我查上月销售额", "好的小张，上月销售额是42", "data_query")
    assert identity is True, f"identity statement must be worth remembering, got {identity}"
    throwaway = decide_write_worthy("上月销售额最高的三个商品", "是A/B/C", "data_query")
    print(f"\n[write-gate] 身份陈述→{identity} / 一次性查询→{throwaway}")


def test_latency_statistics(decision_env):
    # one warm-up, then 10 measured rounds of a one-question decision
    decide_intent("上月销售额")
    latencies = []
    for i in range(10):
        started = time.perf_counter()
        assert decide_intent("上月销售额") is not None
        latencies.append((time.perf_counter() - started) * 1000)
    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    print(f"\n[latency] mean={mean:.1f}ms median={median:.1f}ms "
          f"min={min(latencies):.1f}ms max={max(latencies):.1f}ms")
    # bounded by the client timeout, with generous margin for cold GPU
    assert mean < 2000.0
