"""
app/memory/service.py

Long-term memory service — mem0 OSS single engine, dual instances
(DESIGN_long_term_memory.md):

- FACTS instance (`infer=True`):  LLM extraction + write-time consolidation
  (ADD/UPDATE/DELETE). The rename scenario resolves via UPDATE (supersede),
  verified by the M0 gate spike on mem0ai 0.1.116.
- EPISODIC instance (`infer=False`): raw pre-formatted episode lines stored
  verbatim — zero LLM, append-only.

All contact with mem0 is confined to this module. The read path (recall) runs
in the request path but is millisecond-level local vector search with a
timeout and never raises (fail-open). The write path (write_turn) is
dispatched as a FastAPI background task after the response is sent.

Version note: mem0ai is locked to >=0.1.116,<0.2 — the 0.1.x two-phase
pipeline passes the rename gate; 0.1.117+ pin openai<1.100 (conflicts with
the project's openai 2.x/3.x stack) and 2.x storage is additive-only (old
names keep ranking first). The LLM goes through the OPENAI provider pointed
at the DeepSeek-compatible endpoint because 0.1.116's DeepSeekLLM silently
drops response_format (json_object) which makes deepseek-flash JSON output
unstable. See DESIGN §7.1.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from typing import Any, NamedTuple

# Must run before mem0 is imported anywhere — module-level telemetry switch.
os.environ.setdefault("MEM0_TELEMETRY", "False")

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Chinese fact-extraction prompt (M0 gate validated). NOTE: mem0ai 0.1.x uses
# custom_fact_extraction_prompt as a FULL replacement of the default prompt,
# so the {"facts": [...]} output contract and few-shots must be included —
# a bare instruction makes the LLM answer prose and extraction yields zero.
FACT_EXTRACTION_PROMPT_ZH = """\
你是个人信息整理器：从中文对话中精确抽取值得长期记住的用户事实，组织成独立、简短的事实条目。

重点抽取范围：
1. 身份与称呼：用户自称/被确认的名字、称呼方式；
2. 业务口径：用户明确表达或确认的统计口径（如"销售额按含税算"、常用时间范围、
   "有效订单"的定义）；
3. 稳定偏好：反复出现的表达偏好（如"只看 TOP5""回答要简洁"）。
不抽取：一次性问题的具体数字结论（由情景层负责）、与用户无关的系统信息、
临时的寒暄。
用户给出同一事实的新值时（例如改名），抽取为一条独立的新事实
（如"名叫李四"），不要在事实里混入历史沿革叙述。

示例：
Input: 你好。
Output: {"facts" : []}

Input: 你能查订单吗？
Output: {"facts" : []}

Input: 我叫张三，帮我查上个月销售额。
Output: {"facts" : ["名叫张三"]}

Input: 以后销售额都按含税口径算，只看TOP5就行。
Output: {"facts" : ["销售额按含税口径计算", "查询结果只看TOP5"]}

Input: 我现在改名叫李四了。
Output: {"facts" : ["名叫李四"]}

输出只能是如下 JSON（无事实时 facts 为空数组），不要输出任何其他文字：
{"facts" : ["事实1", "事实2"]}"""

# Episode line content budgets (chars) — mirrors the ChatTurn.answer[:200] habit
_SQL_PREVIEW_CHARS = 120
_ANSWER_PREVIEW_CHARS = 200

_INTENT_TAGS = {"data_query": "数据查询", "smalltalk": "闲聊", "unsafe": "已拒绝"}


class RecallResult(NamedTuple):
    """Formatted long-term memory lines ready for prompt injection."""

    facts: list[str]     # current-valid facts (identity / conventions / preferences)
    episodes: list[str]  # episodic excerpts ("[ts|tag] 问：…｜SQL：…｜答：…")


def format_episode(
    question: str,
    intent: str,
    sql: str | None,
    answer: str | None,
    session_id: str,
    now: datetime | None = None,
) -> str:
    """
    Compose the one-line episodic record stored verbatim via add(infer=False).

    The timestamp leads because half the value of an episodic hit is "when";
    budgets bound the line length (SQL 120 / answer 200 chars). Clarify turns
    have neither SQL nor answer and are tagged 未出结果 by the caller's intent
    plus missing fields.
    """
    ts = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
    tag = _INTENT_TAGS.get(intent, intent)
    parts = [f"[{ts}｜{tag}｜会话{session_id[:8]}] 问：{question}"]
    if sql:
        parts.append(f"SQL：{sql[:_SQL_PREVIEW_CHARS]}")
    if answer:
        parts.append(f"答：{answer[:_ANSWER_PREVIEW_CHARS]}")
    return "｜".join(parts)


class MemoryService:
    """
    Thin orchestration over the two mem0 Memory instances.

    Both public methods NEVER raise (fail-open, DR-L9): any mem0/Qdrant/LLM
    failure degrades to empty recall or a skipped write, logged only.
    """

    def __init__(self, facts_memory: Any, episodic_memory: Any, settings: Settings):
        self._facts = facts_memory
        self._episodic = episodic_memory
        self._settings = settings

    # ------------------------------------------------------------------ read

    def recall(self, question: str, user_id: str | None = None) -> RecallResult:
        """
        Parallel two-leg recall (facts + episodes), each leg independently
        fail-open with a timeout. Local vector search, no LLM call.
        """
        uid = user_id or self._settings.mem0_user_id
        settings = self._settings
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                facts_future = executor.submit(
                    self._safe_leg,
                    self._facts,
                    question,
                    uid,
                    settings.mem0_facts_top_k,
                    settings.mem0_facts_score_threshold,
                )
                episodes_future = executor.submit(
                    self._safe_leg,
                    self._episodic,
                    question,
                    uid,
                    settings.mem0_episodic_top_k,
                    0.0,
                )
                facts = facts_future.result(timeout=settings.mem0_recall_timeout_seconds)
                episodes = episodes_future.result(
                    timeout=settings.mem0_recall_timeout_seconds
                )
            return RecallResult(facts=facts, episodes=episodes)
        except Exception as exc:  # noqa: BLE001 — fail-open by contract (timeout etc.)
            logger.warning("long-term memory recall failed (fail-open): %s", exc)
            return RecallResult(facts=[], episodes=[])

    def _safe_leg(
        self, memory: Any, question: str, user_id: str, top_k: int, score_threshold: float
    ) -> list[str]:
        """One leg, isolated fail-open: its failure empties only this leg."""
        try:
            return self._search_leg(memory, question, user_id, top_k, score_threshold)
        except Exception as exc:  # noqa: BLE001
            logger.warning("long-term memory leg failed (fail-open): %s", exc)
            return []

    @staticmethod
    def _search_leg(
        memory: Any, question: str, user_id: str, top_k: int, score_threshold: float
    ) -> list[str]:
        """One mem0 search, client-side score filter, formatted lines out."""
        hits = memory.search(question, user_id=user_id, limit=top_k)
        results = [
            item["memory"]
            for item in hits.get("results", [])
            if (item.get("score") or 0.0) >= score_threshold and item.get("memory")
        ]
        return results[:top_k]

    def search_with_scores(
        self, question: str, user_id: str | None = None, top_k: int = 5
    ) -> dict[str, list[dict[str, Any]]]:
        """Raw two-leg search with scores — powers the /memory debug endpoint."""
        uid = user_id or self._settings.mem0_user_id
        try:
            facts = self._facts.search(question, user_id=uid, limit=top_k).get(
                "results", []
            )
            episodes = self._episodic.search(question, user_id=uid, limit=top_k).get(
                "results", []
            )
            trim = lambda items: [  # noqa: E731 — small local projection
                {
                    "memory": item.get("memory"),
                    "score": item.get("score"),
                    "created_at": item.get("created_at"),
                }
                for item in items
            ]
            return {"facts": trim(facts), "episodes": trim(episodes)}
        except Exception as exc:  # noqa: BLE001 — fail-open by contract
            logger.warning("long-term memory debug search failed: %s", exc)
            return {"facts": [], "episodes": []}

    # ----------------------------------------------------------------- write

    def write_turn(
        self,
        *,
        question: str,
        intent: str,
        sql: str | None,
        answer: str | None,
        session_id: str,
        trace_id: str | None = None,
        user_id: str | None = None,
        skip_facts: bool = False,
    ) -> None:
        """
        Background-task entry point, runs AFTER the response is sent.

        Facts leg: infer=True — 1~2 deepseek-flash calls (seconds), extracts
        and consolidates user-stated facts. The assistant message is omitted
        when empty so pure-question turns don't feed a blank reply.
        skip_facts=True (write gate said "not memorable", DR-5) skips only
        this leg — the 1~2 LLM calls are what the gate exists to save.
        Episodic leg: infer=False — formatted episode line stored verbatim
        with metadata for future time-decay / per-session cleanup. ALWAYS
        runs: it is the rebuild source for memory_cli and costs no LLM.

        Unsafe turns never reach this method (route skips them); the guard
        here is defense in depth.
        """
        if intent == "unsafe":
            logger.info("long-term memory: skip unsafe turn (trace=%s)", trace_id)
            return
        uid = user_id or self._settings.mem0_user_id

        if skip_facts:
            logger.info(
                "long-term memory: write gate says not memorable — facts leg "
                "skipped, episodic kept (trace=%s)",
                trace_id,
            )
        else:
            try:
                messages = [{"role": "user", "content": question}]
                if answer:
                    messages.append({"role": "assistant", "content": answer})
                result = self._facts.add(messages, user_id=uid, infer=True)
                logger.info(
                    "mem0 facts add: %s",
                    [(r.get("event"), r.get("memory")) for r in result.get("results", [])],
                )
            except Exception as exc:  # noqa: BLE001 — fail-open by contract
                logger.error("mem0 facts write failed (dropped, rebuildable): %s", exc)

        try:
            episode = format_episode(question, intent, sql, answer, session_id)
            self._episodic.add(
                [{"role": "user", "content": episode}],
                user_id=uid,
                infer=False,
                metadata={
                    "session_id": session_id,
                    "trace_id": trace_id,
                    "intent": intent,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
        except Exception as exc:  # noqa: BLE001 — fail-open by contract
            logger.error("mem0 episodic write failed (dropped, rebuildable): %s", exc)


class NullMemoryService:
    """Returned when memory_enabled=false — recall is empty, writes are no-ops."""

    def recall(self, question: str, user_id: str | None = None) -> RecallResult:
        return RecallResult(facts=[], episodes=[])

    def write_turn(self, **_: Any) -> None:
        return None

    def search_with_scores(
        self, question: str, user_id: str | None = None, top_k: int = 5
    ) -> dict[str, list[dict[str, Any]]]:
        return {"facts": [], "episodes": []}


def _build_config(
    settings: Settings, collection: str, history_db: str, extraction_prompt: str | None
) -> dict:
    """mem0 config dict for one Memory instance (facts or episodic)."""
    config: dict[str, Any] = {
        # openai provider (NOT mem0's deepseek): 0.1.116 DeepSeekLLM drops
        # response_format → unstable JSON from the thinking model. The openai
        # provider points at the DeepSeek-compatible endpoint and passes
        # response_format through. Verified by the M0 spike.
        "llm": {
            "provider": "openai",
            "config": {
                "model": settings.deepseek_model,
                "api_key": settings.deepseek_api_key.get_secret_value(),
                "openai_base_url": settings.deepseek_base_url,
                "temperature": 0.0,
            },
        },
        # Reuse the process-wide BGE-M3 singleton (same GPU weights, same
        # vector space as text2sql_schema/text2sql_docs — M0: cosine=1.0).
        "embedder": {
            "provider": "langchain",
            "config": _embedder_config(),
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
                "collection_name": collection,
                "embedding_model_dims": 1024,
            },
        },
        "history_db_path": history_db,
    }
    if extraction_prompt:
        config["custom_fact_extraction_prompt"] = extraction_prompt
    return config


def _embedder_config() -> dict:
    """Import lazily so test/off environments never load the GPU model."""
    from app.rag.embedder import _get_embeddings

    return {"model": _get_embeddings(), "embedding_dims": 1024}


def _silence_mem0_capture_event() -> None:
    """
    0.1.116 defect workaround, confined to this module (the only mem0 contact
    point): telemetry.capture_event() constructs a fresh Posthog client AND
    writes an anonymous-id point on EVERY call — MEM0_TELEMETRY=False only
    sets posthog.disabled after construction, so a long-running server would
    accumulate idle clients/threads per add/search. No data leaves the machine
    either way; this just removes the churn. Revisit on a mem0 upgrade.
    """
    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    import mem0.memory.main as _mem0_main
    import mem0.memory.telemetry as _mem0_telemetry

    _mem0_telemetry.capture_event = _noop
    _mem0_main.capture_event = _noop  # main.py binds the name at import time


@lru_cache(maxsize=1)
def _build_service() -> MemoryService:
    """Construct the real dual-instance service (may raise; never cached on failure)."""
    # Imported here so `import app.memory` stays mem0-free for tests.
    from mem0 import Memory

    _silence_mem0_capture_event()
    settings = get_settings()
    facts = Memory.from_config(
        _build_config(
            settings,
            settings.mem0_facts_collection,
            settings.mem0_facts_history_db,
            FACT_EXTRACTION_PROMPT_ZH,
        )
    )
    episodic = Memory.from_config(
        _build_config(
            settings,
            settings.mem0_episodic_collection,
            settings.mem0_episodic_history_db,
            None,  # raw storage, no extraction prompt used
        )
    )
    return MemoryService(facts, episodic, settings)


def get_memory_service() -> MemoryService | NullMemoryService:
    """
    Cached accessor with fail-open construction.

    Returns the Null service when memory_enabled=false or when construction
    fails (e.g. Qdrant down) — a failed construction is NOT cached, so the
    next call retries. The write path still works through query_run and both
    mem0 stores remain rebuildable from it (DR-L10).
    """
    settings = get_settings()
    if not settings.memory_enabled:
        return NullMemoryService()
    try:
        return _build_service()
    except Exception as exc:  # noqa: BLE001 — fail-open by contract
        logger.warning("mem0 service construction failed (fail-open): %s", exc)
        return NullMemoryService()
