"""
CORRECT node: three-channel error contract (validation / execution / semantic
answer_failure_reason), earlier-attempts injection, and the LLM-failure
regression.
"""

from app.agent.nodes.correct import correct
from app.agent.nodes.generate import SQLGenerationResponse
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
        "app.agent.nodes.correct.get_structured_llm", lambda schema: fake
    )


def _parsed(sql: str = "SELECT 2"):
    return SQLGenerationResponse(
        sql=sql, reasoning="换了一种查法", assumptions=[], confidence=0.8
    )


def _attempt(number: int, failed_sql: str, error: str) -> dict:
    return {
        "attempt_number": number,
        "failed_sql": failed_sql,
        "error_message": error,
        "corrected_sql": None,
    }


# ---------------------------------------------------------------------------
# Three-channel error contract
# ---------------------------------------------------------------------------


def test_validation_error_channel(make_state, monkeypatch):
    fake = FakeStructuredLLM(parsed=_parsed())
    _patch_llm(monkeypatch, fake)

    result = correct(
        make_state(generated_sql="SELECT 1", validation_error="column does not exist")
    )

    assert result["correction_history"][-1]["error_message"] == "column does not exist"
    assert "column does not exist" in fake.calls[0][1]["content"]


def test_execution_error_channel(make_state, monkeypatch):
    fake = FakeStructuredLLM(parsed=_parsed())
    _patch_llm(monkeypatch, fake)

    result = correct(make_state(generated_sql="SELECT 1", execution_error="db gone"))

    assert result["correction_history"][-1]["error_message"] == "db gone"
    assert "db gone" in fake.calls[0][1]["content"]


def test_answer_failure_reason_channel(make_state, monkeypatch):
    fake = FakeStructuredLLM(parsed=_parsed())
    _patch_llm(monkeypatch, fake)

    reason = "查到 0 行；名称匹配串过窄，建议放宽为 LIKE '%星幻%'"
    result = correct(
        make_state(
            generated_sql="SELECT 1", answer_failure_reason=reason, retry_count=1
        )
    )

    record = result["correction_history"][-1]
    assert record["error_message"] == reason        # carried by the history record
    assert reason in fake.calls[0][1]["content"]    # fed to the LLM as the error
    assert result["answer_failure_reason"] is None  # consumed and cleared
    assert result["retry_count"] == 2
    # semantic failures get the change-strategy guidance
    assert "Semantic-failure corrections" in fake.calls[0][0]["content"]


# ---------------------------------------------------------------------------
# Earlier-attempts injection
# ---------------------------------------------------------------------------


def test_earlier_attempts_injected_only_when_present(make_state, monkeypatch):
    fake = FakeStructuredLLM(parsed=_parsed())
    _patch_llm(monkeypatch, fake)

    history = [_attempt(1, "SELECT a FROM t", "查到 0 行；匹配串过窄")]
    result = correct(
        make_state(
            generated_sql="SELECT b FROM t",
            answer_failure_reason="仍未命中，需要换表",
            retry_count=1,
            correction_history=history,
        )
    )

    user_content = fake.calls[0][1]["content"]
    assert "Earlier attempts" in user_content
    assert "SELECT a FROM t" in user_content        # prior attempt rendered
    assert "Previous attempt #2" in user_content    # current attempt listed separately
    assert result["correction_history"][-1]["attempt_number"] == 2

    # no earlier history → the block is omitted entirely
    fake2 = FakeStructuredLLM(parsed=_parsed())
    _patch_llm(monkeypatch, fake2)
    correct(make_state(generated_sql="SELECT 1", validation_error="bad sql"))
    assert "Earlier attempts" not in fake2.calls[0][1]["content"]


# ---------------------------------------------------------------------------
# LLM failure regression
# ---------------------------------------------------------------------------


def test_llm_failure_returns_no_corrected_sql(make_state, monkeypatch):
    _patch_llm(monkeypatch, FakeStructuredLLM(error=RuntimeError("api down")))

    result = correct(
        make_state(generated_sql="SELECT 1", validation_error="bad sql")
    )

    assert result["generated_sql"] is None
    assert result["retry_count"] == 1
    assert result["correction_history"][-1]["error_message"] == "bad sql"
