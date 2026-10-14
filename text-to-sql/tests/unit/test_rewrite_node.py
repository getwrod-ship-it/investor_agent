"""
REWRITE node: passthrough without history, LLM rewrite with history,
fallback to the original question on any failure, and the local decision
gate (self-contained questions in multi-turn sessions pass through).
"""

from app.agent.nodes.rewrite import QuestionRewrite, rewrite
from langchain_core.messages import AIMessage


class FakeStructuredLLM:
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
        "app.agent.nodes.rewrite.get_structured_llm", lambda schema: fake
    )


def test_passthrough_without_history(make_state, monkeypatch):
    # If history is empty the node must NOT call the LLM at all — patch it
    # with an exploding fake to prove the code path is never taken.
    def explode(schema):
        raise AssertionError("get_structured_llm must not be called without history")

    monkeypatch.setattr("app.agent.nodes.rewrite.get_structured_llm", explode)

    result = rewrite(make_state(question="上个月销售额最高的三个商品是什么？"))

    assert result["rewritten_question"] == "上个月销售额最高的三个商品是什么？"


def test_rewrites_followup_with_history(make_state, monkeypatch):
    fake = FakeStructuredLLM(
        parsed=QuestionRewrite(
            rewritten_question="2026年8月销售额最高的三个商品是什么？", changed=True
        )
    )
    _patch_llm(monkeypatch, fake)

    history = [
        {
            "question": "上个月销售额最高的三个商品是什么？",
            "sql": "SELECT goods_name FROM litemall_order GROUP BY 1",
            "answer": "TOP3 是A/B/C",
        }
    ]
    result = rewrite(
        make_state(question="那前个月呢？", chat_history=history)  # type: ignore[arg-type]
    )

    assert result["rewritten_question"] == "2026年8月销售额最高的三个商品是什么？"
    assert len(fake.calls) == 1
    # history must reach the prompt
    system_content = fake.calls[0][0]["content"]
    assert "litemall_order" in system_content


def test_falls_back_to_original_on_llm_error(make_state, monkeypatch):
    _patch_llm(monkeypatch, FakeStructuredLLM(error=RuntimeError("api down")))

    history = [{"question": "上月销量", "sql": "SELECT 1", "answer": "42"}]
    result = rewrite(make_state(question="那前个月呢？", chat_history=history))  # type: ignore[arg-type]

    assert result["rewritten_question"] == "那前个月呢？"


def test_falls_back_on_empty_rewrite(make_state, monkeypatch):
    _patch_llm(
        monkeypatch,
        FakeStructuredLLM(parsed=QuestionRewrite(rewritten_question="   ", changed=True)),
    )

    history = [{"question": "上月销量", "sql": "SELECT 1", "answer": "42"}]
    result = rewrite(make_state(question="那前个月呢？", chat_history=history))  # type: ignore[arg-type]

    assert result["rewritten_question"] == "那前个月呢？"


# --------------------------------------------------------------------------
# Local decision gate (DESIGN_local_decision_model.md DR-6)
# --------------------------------------------------------------------------


def test_gate_passes_selfcontained_question_through(make_state, monkeypatch):
    # multi-turn session + the gate says self-contained → no LLM call at all
    def explode(schema):
        raise AssertionError("get_structured_llm must not be called when the gate passes through")

    monkeypatch.setattr("app.agent.nodes.rewrite.get_structured_llm", explode)
    monkeypatch.setattr(
        "app.agent.nodes.rewrite.decide_rewrite_needed", lambda q, h: False
    )

    history = [{"question": "上月销量", "sql": "SELECT 1", "answer": "42"}]
    result = rewrite(
        make_state(question="查一下2024年订单总数", chat_history=history)  # type: ignore[arg-type]
    )

    assert result["rewritten_question"] == "查一下2024年订单总数"


def test_gate_none_falls_back_to_llm(make_state, monkeypatch):
    # service down / gate off → None → the DeepSeek rewrite runs as before
    monkeypatch.setattr(
        "app.agent.nodes.rewrite.decide_rewrite_needed", lambda q, h: None
    )
    fake = FakeStructuredLLM(
        parsed=QuestionRewrite(
            rewritten_question="2024年订单总数是多少？", changed=False
        )
    )
    _patch_llm(monkeypatch, fake)

    history = [{"question": "上月销量", "sql": "SELECT 1", "answer": "42"}]
    result = rewrite(
        make_state(question="查一下2024年订单总数", chat_history=history)  # type: ignore[arg-type]
    )

    assert result["rewritten_question"] == "2024年订单总数是多少？"
    assert len(fake.calls) == 1


def test_gate_not_consulted_without_history(make_state, monkeypatch):
    # first-turn passthrough must stay in front of the gate
    consulted: list[str] = []

    def _spy(question, history):
        consulted.append(question)
        return False

    monkeypatch.setattr("app.agent.nodes.rewrite.decide_rewrite_needed", _spy)

    result = rewrite(make_state(question="上个月销售额最高的三个商品是什么？"))

    assert result["rewritten_question"] == "上个月销售额最高的三个商品是什么？"
    assert consulted == []
