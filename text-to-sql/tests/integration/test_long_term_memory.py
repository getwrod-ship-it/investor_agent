"""
REAL end-to-end acceptance for the long-term memory layer (mem0 0.1.116 +
DeepSeek + Qdrant + local BGE-M3). Opt-in only — skipped unless both hold:

    MEM0_IT=1                     (explicit opt-in)
    DEEPSEEK key present in .env  (conftest injects a FAKE key into the env,
                                   so the real key is parsed from .env here)

Run from the project root with Qdrant up (docker compose up -d qdrant):

    MEM0_IT=1 .venv/Scripts/python.exe -m pytest tests/integration/test_long_term_memory.py -v -s

Scenarios (DESIGN_long_term_memory.md §8.3):
1. Rename gold standard: 我叫张三→李四→王五, recall returns ONLY 王五
2. Convention across sessions: 含税口径 fact is recalled for a related question
3. Episodic recall: a past data question's episode is retrieved verbatim
4. Latency: recall P95 well under budget
"""

import os
import time
from contextlib import suppress
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("MEM0_IT") != "1",
    reason="opt-in only: set MEM0_IT=1 (requires real DeepSeek key + Qdrant + GPU model)",
)

# dedicated IT user — everything is scoped to it and wiped afterwards
IT_USER = "mem0-it-user"


def _real_deepseek_key() -> str:
    """Parse the real key from .env (conftest overrides the env var with a fake)."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("DEEPSEEK_API_KEY="):
            key = line.split("=", 1)[1].strip()
            if key and not key.startswith("your"):
                return key
    pytest.skip("no real DEEPSEEK_API_KEY in .env")


@pytest.fixture(scope="module")
def service():
    """Real dual-instance MemoryService against the production collections."""
    from types import SimpleNamespace

    from app.config import get_settings
    from app.memory.service import FACT_EXTRACTION_PROMPT_ZH, MemoryService, _build_config
    from mem0 import Memory
    from pydantic import SecretStr

    real = SimpleNamespace(
        deepseek_model=get_settings().deepseek_model,
        deepseek_api_key=SecretStr(_real_deepseek_key()),
        deepseek_base_url=get_settings().deepseek_base_url,
        qdrant_host=get_settings().qdrant_host,
        qdrant_port=get_settings().qdrant_port,
    )
    facts = Memory.from_config(
        _build_config(real, get_settings().mem0_facts_collection,
                      get_settings().mem0_facts_history_db, FACT_EXTRACTION_PROMPT_ZH)
    )
    episodic = Memory.from_config(
        _build_config(real, get_settings().mem0_episodic_collection,
                      get_settings().mem0_episodic_history_db, None)
    )
    svc = MemoryService(facts, episodic, get_settings())

    # clean slate for the IT user only (prod collections untouched for others)
    for store in (facts, episodic):
        with suppress(Exception):
            store.delete_all(user_id=IT_USER)

    yield svc

    for store in (facts, episodic):
        with suppress(Exception):
            store.delete_all(user_id=IT_USER)


def test_rename_gold_standard(service):
    for q, a in [
        ("我叫张三", "好的，张三！"),
        ("我现在改名叫李四了", "好的，李四！"),
        ("我又改名了，现在叫王五", "好的，王五！"),
    ]:
        service.write_turn(
            question=q, intent="smalltalk", sql=None, answer=a,
            session_id="it-sess-rename", trace_id="it", user_id=IT_USER,
        )

    result = service.recall("我叫什么名字", user_id=IT_USER)
    joined = " ".join(result.facts)
    assert "王五" in joined, f"facts={result.facts}"
    assert "张三" not in joined and "李四" not in joined, f"stale names leaked: {result.facts}"


def test_convention_fact_recalled_across_sessions(service):
    service.write_turn(
        question="以后销售额都按含税口径算",
        intent="smalltalk",
        sql=None,
        answer="好的，已按含税口径记下。",
        session_id="it-sess-conv",
        trace_id="it",
        user_id=IT_USER,
    )

    result = service.recall("销售额怎么算", user_id=IT_USER)
    assert any("含税" in f for f in result.facts), f"facts={result.facts}"


def test_episodic_recalled_verbatim(service):
    service.write_turn(
        question="上个月退款金额TOP3的商品是哪些？",
        intent="data_query",
        sql="SELECT g.name FROM aftersale a JOIN goods g ... LIMIT 3",
        answer="TOP1 是居家布艺沙发，退款 1.2 万。",
        session_id="it-sess-epi",
        trace_id="it",
        user_id=IT_USER,
    )

    result = service.recall("上次那个退款排名的结论", user_id=IT_USER)
    assert any("退款金额TOP3" in e for e in result.episodes), f"episodes={result.episodes}"


def test_recall_latency_p95(service):
    service.recall("预热", user_id=IT_USER)  # warm the embedding/GPU path
    durations = []
    for i in range(20):
        start = time.perf_counter()
        service.recall(f"延迟测量问题 {i}", user_id=IT_USER)
        durations.append(time.perf_counter() - start)
    durations.sort()
    p95 = durations[int(len(durations) * 0.95) - 1]
    assert p95 < 0.2, f"recall P95={p95:.3f}s (budget: design targets 0.1s)"
