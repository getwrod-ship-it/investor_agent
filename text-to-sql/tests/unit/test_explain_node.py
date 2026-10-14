"""
EXPLAIN node: three-way answerability verdict (answer / retry / give_up),
final-attempt safety net, fail-open paths, and the legacy kill-switch path.
"""

import pytest
from app.agent.nodes.explain import ExplainResponse, explain
from app.config import get_settings
from langchain_core.messages import AIMessage


class FakeStructuredLLM:
    """Mimics get_structured_llm(...).invoke() → {"parsed": ..., "raw": ...}."""

    def __init__(self, parsed=None, raw_content: str = "", error: Exception | None = None):
        self.parsed = parsed
        self.raw_content = raw_content
        self.error = error
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return {"parsed": self.parsed, "raw": AIMessage(content=self.raw_content)}


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


def _patch_structured(monkeypatch, fake: FakeStructuredLLM):
    monkeypatch.setattr(
        "app.agent.nodes.explain.get_structured_llm", lambda schema: fake
    )


def _patch_chat(monkeypatch, fake: FakeChatLLM):
    monkeypatch.setattr(
        "app.agent.nodes.explain.get_chat_llm", lambda temperature=0: fake
    )


def _execution(rows=None, row_count=None):
    rows = rows if rows is not None else [{"goods_name": "星幻Pro", "cnt": 3}]
    return {
        "rows": rows,
        "row_count": row_count if row_count is not None else len(rows),
        "execution_time_ms": 5.0,
    }


def _attempt(number: int, failed_sql: str, error: str) -> dict:
    return {
        "attempt_number": number,
        "failed_sql": failed_sql,
        "error_message": error,
        "corrected_sql": None,
    }


@pytest.fixture
def flag_off(monkeypatch):
    """Disable the answerability flag for one test; settings cache restored after."""
    monkeypatch.setenv("EXPLAIN_ANSWERABILITY_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Three-way verdict
# ---------------------------------------------------------------------------


def test_answer_verdict_sets_final_explanation(make_state, monkeypatch):
    parsed = ExplainResponse(verdict="answer", answer="上个月卖出 3 件。", reason=None)
    fake = FakeStructuredLLM(parsed=parsed)
    _patch_structured(monkeypatch, fake)

    result = explain(make_state(execution_result=_execution()))

    assert result["final_explanation"] == "上个月卖出 3 件。"
    assert result["answer_failure_reason"] is None


def test_retry_verdict_requests_semantic_retry(make_state, monkeypatch):
    parsed = ExplainResponse(
        verdict="retry", answer=None, reason="查到 0 行；名称匹配串过窄，建议放宽为 LIKE '%星幻%'"
    )
    fake = FakeStructuredLLM(parsed=parsed)
    _patch_structured(monkeypatch, fake)

    result = explain(
        make_state(execution_result=_execution(rows=[], row_count=0), retry_count=0)
    )

    assert result["final_explanation"] is None
    assert result["answer_failure_reason"] == "查到 0 行；名称匹配串过窄，建议放宽为 LIKE '%星幻%'"


def test_give_up_verdict_terminates_without_waiting_for_quota(make_state, monkeypatch):
    # LLM-autonomous termination: give_up on a non-final round (1 < max_retries=3)
    parsed = ExplainResponse(
        verdict="give_up", answer="试过的查法都未命中，库里可能没有该商品。", reason=None
    )
    fake = FakeStructuredLLM(parsed=parsed)
    _patch_structured(monkeypatch, fake)

    result = explain(
        make_state(execution_result=_execution(rows=[], row_count=0), retry_count=1)
    )

    assert result["final_explanation"] == "试过的查法都未命中，库里可能没有该商品。"
    assert result["answer_failure_reason"] is None


# ---------------------------------------------------------------------------
# Prompt assembly: discipline / attempts / final attempt
# ---------------------------------------------------------------------------


def test_first_round_prompt_has_discipline_and_no_attempts_block(make_state, monkeypatch):
    parsed = ExplainResponse(verdict="answer", answer="ok")
    fake = FakeStructuredLLM(parsed=parsed)
    _patch_structured(monkeypatch, fake)

    explain(make_state(execution_result=_execution(), correction_history=[]))

    system_content = fake.calls[0][0].content
    assert "第一轮" in system_content
    assert "禁止 give_up" in system_content
    # the attempts block header (## 此前的尝试记录) — the bare phrase also
    # appears in the base give_up criteria, so assert on the block header
    assert "## 此前的尝试记录" not in system_content
    assert "最终轮要求" not in system_content


def test_attempts_history_injected_into_system_prompt(make_state, monkeypatch):
    parsed = ExplainResponse(verdict="answer", answer="ok")
    fake = FakeStructuredLLM(parsed=parsed)
    _patch_structured(monkeypatch, fake)

    history = [
        _attempt(1, "SELECT 1", "查到 0 行；匹配串过窄"),
        _attempt(2, "SELECT 2", "查到 0 行；可能查错了表"),
    ]
    explain(
        make_state(execution_result=_execution(), correction_history=history, retry_count=2)
    )

    system_content = fake.calls[0][0].content
    assert "此前的尝试记录" in system_content
    assert "# 尝试 1" in system_content
    assert "# 尝试 2" in system_content
    assert "SELECT 1" in system_content


def test_final_attempt_forbids_retry_and_forces_terminal_answer(make_state, monkeypatch):
    # quota exhausted (3 >= max_retries=3): even a retry verdict is forced terminal
    parsed = ExplainResponse(verdict="retry", answer=None, reason="还想再换种查法")
    fake = FakeStructuredLLM(parsed=parsed)
    _patch_structured(monkeypatch, fake)

    result = explain(make_state(execution_result=_execution(), retry_count=3))

    system_content = fake.calls[0][0].content
    assert "最终轮要求" in system_content
    assert "禁止选择 retry" in system_content
    # force-through: no model answer → static fallback, never a retry request
    assert result["final_explanation"]
    assert "生成回答失败" in result["final_explanation"]
    assert result["answer_failure_reason"] is None


# ---------------------------------------------------------------------------
# Fail-open paths
# ---------------------------------------------------------------------------


def test_llm_exception_falls_back_to_static_text(make_state, monkeypatch):
    _patch_structured(monkeypatch, FakeStructuredLLM(error=RuntimeError("api down")))

    result = explain(make_state(execution_result=_execution()))

    assert "生成回答失败" in result["final_explanation"]
    assert result["answer_failure_reason"] is None


def test_parse_failure_uses_raw_content_as_answer(make_state, monkeypatch):
    fake = FakeStructuredLLM(parsed=None, raw_content="原始文本兜底回答")
    _patch_structured(monkeypatch, fake)

    result = explain(make_state(execution_result=_execution()))

    assert result["final_explanation"] == "原始文本兜底回答"
    assert result["answer_failure_reason"] is None


def test_parse_failure_salvages_answer_from_valid_json(make_state, monkeypatch):
    # wrapped output that json_mode's pydantic step rejected but loads fine
    fake = FakeStructuredLLM(
        parsed=None,
        raw_content='{"verdict": "answer", "answer": "从包装里救出的回答", "reason": null}',
    )
    _patch_structured(monkeypatch, fake)

    result = explain(make_state(execution_result=_execution()))

    assert result["final_explanation"] == "从包装里救出的回答"
    assert result["answer_failure_reason"] is None


def test_parse_failure_never_dumps_broken_json_at_user(make_state, monkeypatch):
    # unescaped inner quotes break the JSON — the user must get the static
    # fallback, never the raw JSON blob
    fake = FakeStructuredLLM(
        parsed=None,
        raw_content='{"verdict": "answer", "answer": "含"未转义引号"的回答"}',
    )
    _patch_structured(monkeypatch, fake)

    result = explain(make_state(execution_result=_execution()))

    assert "生成回答失败" in result["final_explanation"]
    assert not result["final_explanation"].lstrip().startswith("{")
    assert result["answer_failure_reason"] is None


# ---------------------------------------------------------------------------
# Kill-switch legacy path
# ---------------------------------------------------------------------------


def test_flag_off_uses_legacy_free_text_path(make_state, monkeypatch, flag_off):
    chat = FakeChatLLM(content="旧版纯文本回答")

    def explode(schema):
        raise AssertionError("get_structured_llm must not be called when the flag is off")

    _patch_chat(monkeypatch, chat)
    monkeypatch.setattr("app.agent.nodes.explain.get_structured_llm", explode)

    # retry_count exhausted + history present — legacy path ignores all of it
    result = explain(
        make_state(
            execution_result=_execution(),
            retry_count=3,
            correction_history=[_attempt(1, "SELECT 1", "查到 0 行")],
        )
    )

    assert result["final_explanation"] == "旧版纯文本回答"
    assert result["answer_failure_reason"] is None
    assert len(chat.calls) == 1
    system_content = chat.calls[0][0].content
    assert "You are a data analyst" in system_content
    assert "此前的尝试记录" not in system_content


# ---------------------------------------------------------------------------
# Prompt completeness
# ---------------------------------------------------------------------------


def test_user_prompt_contains_question_sql_rows_and_format_instructions(
    make_state, monkeypatch
):
    parsed = ExplainResponse(verdict="answer", answer="ok")
    fake = FakeStructuredLLM(parsed=parsed)
    _patch_structured(monkeypatch, fake)

    rows = [{"id": i, "goods_name": f"item{i}"} for i in range(15)]
    explain(
        make_state(
            question="销量前三是哪些商品",
            generated_sql="SELECT goods_name FROM orders",
            execution_result=_execution(rows=rows, row_count=15),
        )
    )

    user_content = fake.calls[0][1].content
    assert "销量前三是哪些商品" in user_content
    assert "SELECT goods_name FROM orders" in user_content
    assert "15 行" in user_content
    assert "not shown" in user_content          # 10-row preview truncation
    assert "item9" in user_content
    assert "item10" not in user_content         # rows beyond the preview are cut
    assert "verdict" in user_content            # json format instructions injected
