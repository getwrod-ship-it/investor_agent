"""
Unit tests for app/memory/service.py — MemoryService / NullMemoryService /
format_episode. All mem0 contact is faked; no Qdrant / LLM / GPU needed.

Settings come from the conftest test env (MEMORY_ENABLED=false) which is fine
here: MemoryService reads only the plain knobs (top_k / threshold / user_id).
"""

from datetime import datetime

from app.config import get_settings
from app.memory.service import (
    FACT_EXTRACTION_PROMPT_ZH,
    MemoryService,
    NullMemoryService,
    format_episode,
    get_memory_service,
)


class FakeMemory:
    """Records search/add calls; returns canned results or raises on demand."""

    def __init__(self, results: list[dict] | None = None, raise_on: set[str] | None = None):
        self.searches: list[tuple] = []
        self.adds: list[dict] = []
        self.results = results or []
        self.raise_on = raise_on or set()

    def search(self, query: str, user_id: str | None = None, limit: int = 10) -> dict:
        self.searches.append((query, user_id, limit))
        if "search" in self.raise_on:
            raise RuntimeError("search boom")
        return {"results": self.results}  # ignores limit on purpose (truncation is ours)

    def add(self, messages, user_id=None, infer=True, metadata=None) -> dict:
        self.adds.append(
            {"messages": messages, "user_id": user_id, "infer": infer, "metadata": metadata}
        )
        if "add" in self.raise_on:
            raise RuntimeError("add boom")
        return {"results": [{"event": "ADD", "memory": "x"}]}


def make_service(
    facts: FakeMemory | None = None, episodic: FakeMemory | None = None
) -> MemoryService:
    return MemoryService(facts or FakeMemory(), episodic or FakeMemory(), get_settings())


# ------------------------------------------------------------------- recall


def test_recall_formats_and_filters_by_threshold():
    # conftest settings: mem0_facts_score_threshold=0.5, top_k=5/2
    facts = FakeMemory(
        results=[
            {"memory": "名叫王五", "score": 0.61},
            {"memory": "低分事实", "score": 0.30},   # below 0.5 → dropped
            {"memory": "销售额按含税口径计算", "score": 0.55},
        ]
    )
    episodic = FakeMemory(
        results=[
            {"memory": "[2026-10-13|数据查询|会话a] 问：…", "score": 0.42},
        ]
    )
    service = make_service(facts, episodic)

    result = service.recall("我叫什么名字")

    assert result.facts == ["名叫王五", "销售额按含税口径计算"]
    assert len(result.episodes) == 1
    # user_id defaults to the settings value
    assert facts.searches[0][1] == get_settings().mem0_user_id
    assert episodic.searches[0][1] == get_settings().mem0_user_id


def test_recall_truncates_to_top_k():
    many = [{"memory": f"事实{i}", "score": 0.9} for i in range(7)]
    service = make_service(FakeMemory(results=many), FakeMemory(results=many))

    result = service.recall("问题")

    assert len(result.facts) == get_settings().mem0_facts_top_k  # 5
    assert len(result.episodes) == get_settings().mem0_episodic_top_k  # 2


def test_recall_leg_failure_empties_only_that_leg():
    facts = FakeMemory(raise_on={"search"})
    episodic = FakeMemory(results=[{"memory": "片段", "score": 0.7}])

    result = make_service(facts, episodic).recall("问题")

    assert result.facts == []
    assert result.episodes == ["片段"]


# -------------------------------------------------------------- write_turn


def test_write_turn_data_query_feeds_both_legs():
    facts, episodic = FakeMemory(), FakeMemory()

    make_service(facts, episodic).write_turn(
        question="上个月退款金额TOP3？",
        intent="data_query",
        sql="SELECT ... LIMIT 3",
        answer="TOP1 是沙发",
        session_id="sess-12345678",
        trace_id="t-1",
    )

    # facts leg: user+assistant pair through the extraction pipeline
    facts_add = facts.adds[0]
    assert facts_add["infer"] is True
    assert facts_add["user_id"] == get_settings().mem0_user_id
    assert facts_add["messages"] == [
        {"role": "user", "content": "上个月退款金额TOP3？"},
        {"role": "assistant", "content": "TOP1 是沙发"},
    ]
    # episodic leg: verbatim episode line, zero LLM, with metadata
    epi_add = episodic.adds[0]
    assert epi_add["infer"] is False
    assert epi_add["messages"] == [{"role": "user", "content": epi_add["messages"][0]["content"]}]
    line = epi_add["messages"][0]["content"]
    assert "问：上个月退款金额TOP3？" in line
    assert "SQL：SELECT ... LIMIT 3" in line
    assert "答：TOP1 是沙发" in line
    assert "数据查询" in line
    assert epi_add["metadata"]["session_id"] == "sess-12345678"
    assert epi_add["metadata"]["trace_id"] == "t-1"
    assert epi_add["metadata"]["intent"] == "data_query"
    assert "created_at" in epi_add["metadata"]


def test_write_turn_without_answer_omits_assistant_message():
    facts, episodic = FakeMemory(), FakeMemory()

    make_service(facts, episodic).write_turn(
        question="那个报错的问题",
        intent="data_query",
        sql=None,
        answer=None,  # clarify turn: nothing to attribute
        session_id="sess-x",
    )

    assert facts.adds[0]["messages"] == [{"role": "user", "content": "那个报错的问题"}]
    # the episode is still recorded (a failed ask is a real event)
    assert len(episodic.adds) == 1
    assert "问：那个报错的问题" in episodic.adds[0]["messages"][0]["content"]


def test_write_turn_skips_unsafe():
    facts, episodic = FakeMemory(), FakeMemory()

    make_service(facts, episodic).write_turn(
        question="删库", intent="unsafe", sql=None, answer="拒绝", session_id="s"
    )

    assert facts.adds == [] and episodic.adds == []


def test_write_turn_leg_failure_does_not_block_the_other_leg():
    facts, episodic = FakeMemory(raise_on={"add"}), FakeMemory()

    make_service(facts, episodic).write_turn(
        question="q", intent="data_query", sql=None, answer="a", session_id="s"
    )

    assert len(facts.adds) == 1   # attempted (and raised) …
    assert len(episodic.adds) == 1  # … but the episodic leg still wrote


def test_write_turn_skip_facts_skips_only_the_facts_leg():
    # write gate said "not memorable" (DR-5): no LLM extraction, but the
    # episodic trace is still recorded — it is the rebuild source
    facts, episodic = FakeMemory(), FakeMemory()

    make_service(facts, episodic).write_turn(
        question="上月销售额",
        intent="data_query",
        sql="SELECT 1",
        answer="42",
        session_id="s",
        skip_facts=True,
    )

    assert facts.adds == []
    assert len(episodic.adds) == 1
    assert "问：上月销售额" in episodic.adds[0]["messages"][0]["content"]


# ----------------------------------------------------------- episode format


def test_format_episode_budgets_and_fields():
    long_sql = "S" * 300
    long_answer = "A" * 400
    line = format_episode(
        "问句", "data_query", long_sql, long_answer, "sess-abcdef123456",
        now=datetime(2026, 10, 13, 14, 32),
    )
    # session marker = first 8 chars of the session id
    assert line.startswith("[2026-10-13 14:32｜数据查询｜会话sess-abc]")
    assert f"SQL：{'S' * 120}" in line and "S" * 121 not in line
    assert f"答：{'A' * 200}" in line and "A" * 201 not in line


def test_format_episode_omits_missing_parts():
    line = format_episode("闲聊句", "smalltalk", None, None, "s", now=datetime(2026, 1, 1))
    assert line == "[2026-01-01 00:00｜闲聊｜会话s] 问：闲聊句"


# ---------------------------------------------------- null service / factory


def test_null_service_is_inert():
    null = NullMemoryService()
    result = null.recall("x")
    assert result.facts == [] and result.episodes == []
    null.write_turn(question="q")  # no-op, no raise
    assert null.search_with_scores("x") == {"facts": [], "episodes": []}


def test_get_memory_service_disabled_returns_null():
    # conftest sets MEMORY_ENABLED=false
    assert isinstance(get_memory_service(), NullMemoryService)


def test_extraction_prompt_carries_json_contract():
    # 0.1.x replaces the whole default prompt — the contract must be built in
    assert '{"facts"' in FACT_EXTRACTION_PROMPT_ZH
    assert "名叫李四" in FACT_EXTRACTION_PROMPT_ZH
