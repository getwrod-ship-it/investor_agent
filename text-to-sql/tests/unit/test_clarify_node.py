"""
CLARIFY node: three dispatch paths with fail-open static fallbacks.
"""

from app.agent.nodes.clarify import (
    AMBIGUOUS_FALLBACK,
    INFRA_FAILURE_MESSAGE,
    RETRY_FALLBACK_TEMPLATE,
    _looks_like_infra_failure,
    clarify,
)
from langchain_core.messages import AIMessage


class FakeChatLLM:
    """Mimics get_chat_llm(...).invoke() → AIMessage."""

    def __init__(self, content: str = "", error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return AIMessage(content=self.content)


def _patch_llm(monkeypatch, fake: FakeChatLLM):
    monkeypatch.setattr("app.agent.nodes.clarify.get_chat_llm", lambda temperature=0: fake)


def _attempt(number: int, failed_sql: str, error: str) -> dict:
    return {
        "attempt_number": number,
        "failed_sql": failed_sql,
        "error_message": error,
        "corrected_sql": None,
    }


# ---------------------------------------------------------------------------
# Path 1: ambiguous → LLM-generated follow-up question
# ---------------------------------------------------------------------------


def test_ambiguous_generates_followup_question(make_state, monkeypatch):
    fake = FakeChatLLM(content="想按哪个口径看'最好卖'——按销售额还是按销量？")
    _patch_llm(monkeypatch, fake)

    state = make_state(
        intent="ambiguous",
        chat_history=[
            {"question": "上个月销量如何？", "sql": "SELECT 1", "answer": "不错"}
        ],
    )
    result = clarify(state)

    assert result["needs_clarification"] is True
    assert result["clarify_reason"] == "ambiguous"
    assert "口径" in result["clarification_message"]
    assert result["final_explanation"] is None
    # prompt carries the question and the session history
    assert len(fake.calls) == 1
    system_content = fake.calls[0][0].content
    user_content = fake.calls[0][1].content
    assert "上个月销量如何" in system_content
    assert "上个月销售额最高的三个商品" in user_content


def test_ambiguous_falls_back_when_llm_fails(make_state, monkeypatch):
    _patch_llm(monkeypatch, FakeChatLLM(error=RuntimeError("api down")))

    result = clarify(make_state(intent="ambiguous"))

    assert result["clarification_message"] == AMBIGUOUS_FALLBACK
    assert result["clarify_reason"] == "ambiguous"


# ---------------------------------------------------------------------------
# Path 2: infra failure → static message, NO LLM call
# ---------------------------------------------------------------------------


def test_infra_failure_returns_static_message_without_llm(make_state, monkeypatch):
    fake = FakeChatLLM(content="不该被调用")
    _patch_llm(monkeypatch, fake)

    history = [
        _attempt(1, "", "No SQL was generated"),
        _attempt(2, "", "No SQL was generated"),
        _attempt(3, "", "No SQL was generated"),
    ]
    result = clarify(make_state(correction_history=history, retry_count=3))

    assert result["clarification_message"] == INFRA_FAILURE_MESSAGE
    assert result["clarify_reason"] == "infra_failure"
    assert len(fake.calls) == 0  # model service is down — must not call the LLM


def test_looks_like_infra_failure_rules():
    all_empty = [_attempt(1, "", "No SQL was generated")]
    mixed_sql = [_attempt(1, "SELECT oops", "syntax error"), _attempt(2, "", "No SQL was generated")]
    other_error = [_attempt(1, "", "something else")]

    assert _looks_like_infra_failure({"correction_history": all_empty}) is True
    assert _looks_like_infra_failure({"correction_history": mixed_sql}) is False
    assert _looks_like_infra_failure({"correction_history": other_error}) is False
    assert _looks_like_infra_failure({"correction_history": []}) is False


# ---------------------------------------------------------------------------
# Path 3: retry exhausted → LLM diagnosis with Chinese fallback
# ---------------------------------------------------------------------------


def test_retry_exhausted_diagnoses_from_history(make_state, monkeypatch):
    fake = FakeChatLLM(content="三次失败都是因为列名不存在，建议换用 xxx 表。")
    _patch_llm(monkeypatch, fake)

    history = [
        _attempt(1, "SELECT price FROM goods", "Unknown column 'price'"),
        _attempt(2, "SELECT retail_price FROM goods", "Unknown column 'retail_price'"),
    ]
    state = make_state(correction_history=history, retry_count=3)
    result = clarify(state)

    assert result["clarify_reason"] == "retry_exhausted"
    assert result["clarification_message"].startswith("三次失败")
    # diagnosis prompt carries the real attempts and errors
    user_content = fake.calls[0][1].content
    assert "SELECT price FROM goods" in user_content
    assert "Unknown column 'price'" in user_content


def test_retry_exhausted_falls_back_when_llm_fails(make_state, monkeypatch):
    _patch_llm(monkeypatch, FakeChatLLM(error=RuntimeError("api down")))

    history = [_attempt(1, "SELECT oops", "syntax error")]
    result = clarify(
        make_state(
            correction_history=history,
            retry_count=3,
            execution_error="syntax error near 'oops'",
        )
    )

    assert result["clarify_reason"] == "retry_exhausted"
    assert result["clarification_message"] == RETRY_FALLBACK_TEMPLATE.format(
        retry_count=3, last_error="syntax error near 'oops'"
    )
