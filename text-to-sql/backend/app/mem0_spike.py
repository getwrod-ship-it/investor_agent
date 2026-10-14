"""
app/mem0_spike.py

M0 版本门禁 spike（DESIGN_long_term_memory.md §9.1）——用真实 DeepSeek key +
Qdrant 裸测锁定版 mem0ai 的事实合并行为，先于一切业务代码定版本。

当前锁定候选：mem0ai 0.1.116（0.1.x 线最新且 openai 约束宽松 >=1.33.0 的版本，
与项目 openai 2.x/3.x 栈共存；0.1.117+ 钉死 openai<1.100 冲突，2.x 线存储纯加性
改名门禁不过——见 DESIGN 文档 §7.1 决策树与回填结论）。

验证内容：
1. 向量一致性：mem0 langchain embedder（复用 BGE-M3 单例）与现有 embed 路径
   余弦相似度 ≈ 1.0（>0.999）。
2. 改名金标准：连喂"我叫张三/李四/王五" → search("我叫什么名字")
   → 只回王五（0.1.x 双阶段管线：抽取 + UPDATE 取代）。
   分别跑【默认 prompt】与【中文 custom_fact_extraction_prompt】两个变体
   （update 决策 prompt 两者都保持官方默认，其 JSON 语义契约不汉化）。

用法（项目根，Qdrant 需在跑）：
    PYTHONPATH=backend .venv/Scripts/python.exe -m app.mem0_spike

跑完自动清理 spike 集合（前缀 text2sql_spike_*）与临时 history db。
"""

import os
import shutil
import tempfile

# 关闭 mem0 遥测（必须先于 mem0 导入）
os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import Memory  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.rag.embedder import _get_embeddings  # noqa: E402

SPIKE_PREFIX = "text2sql_spike"

# 与 DESIGN §4.6 一致的生产候选抽取 prompt（update 决策 prompt 用官方默认）。
# 注意：0.1.x 的 custom_fact_extraction_prompt 是整体替换默认 prompt，必须自带
# {"facts": [...]} 输出契约（含 few-shot），否则 LLM 返回散文导致解析为 0 条。
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


def _clean_collections(client: QdrantClient) -> None:
    for col in client.get_collections().collections:
        if col.name.startswith(SPIKE_PREFIX):
            client.delete_collection(col.name)
            print(f"  [cleanup] deleted collection {col.name}")


def _build_config(collection: str, history_db: str, extraction_prompt: str | None) -> dict:
    settings = get_settings()
    return {
        # 注意：用 openai provider 而非 deepseek——0.1.116 的 DeepSeekLLM 会丢弃
        # response_format（json_object），deepseek-flash 思考模型没有 json 约束时
        # 输出不稳定（spike 实测：update 阶段 JSON 解析偶发失败）。openai provider
        # 指向 DeepSeek 兼容端点则正确透传。
        "llm": {
            "provider": "openai",
            "config": {
                "model": settings.deepseek_model,
                "api_key": settings.deepseek_api_key.get_secret_value(),
                "openai_base_url": settings.deepseek_base_url,
                "temperature": 0.0,
            },
        },
        # 复用进程内 BGE-M3 单例（DR-L8）——同一份 GPU 权重、同一向量空间
        "embedder": {
            "provider": "langchain",
            "config": {"model": _get_embeddings(), "embedding_dims": 1024},
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
        **({"custom_fact_extraction_prompt": extraction_prompt} if extraction_prompt else {}),
    }


def _check_vector_consistency() -> None:
    print("\n=== 1. 向量一致性校验 ===")
    tmp_db = os.path.join(tempfile.mkdtemp(prefix="mem0-spike-"), "h.db")
    mem = Memory.from_config(_build_config(f"{SPIKE_PREFIX}_vec", tmp_db, None))
    text = "销售额按含税口径计算，用户常用时间范围是最近30天"
    v_mem0 = mem.embedding_model.embed(text, "add")
    v_direct = _get_embeddings().embed_query(text)
    dot = sum(a * b for a, b in zip(v_mem0, v_direct, strict=True))
    norm = (sum(a * a for a in v_mem0) ** 0.5) * (sum(b * b for b in v_direct) ** 0.5)
    cosine = dot / norm
    print(f"  dims={len(v_mem0)} cosine={cosine:.6f}")
    assert cosine > 0.999, f"向量不一致: cosine={cosine}"
    print("  PASS — mem0 langchain embedder 与现有路径同向量空间")
    shutil.rmtree(os.path.dirname(tmp_db), ignore_errors=True)


def _rename_scenario(label: str, collection: str, extraction_prompt: str | None) -> None:
    print(f"\n=== 2. 改名金标准 [{label}] ===")
    tmp_dir = tempfile.mkdtemp(prefix="mem0-spike-")
    mem = Memory.from_config(
        _build_config(collection, os.path.join(tmp_dir, "h.db"), extraction_prompt)
    )
    user_id = "spike-user"

    turns = [
        ("我叫张三", "好的，张三！"),
        ("我现在改名叫李四了", "好的，以后就叫你李四了！"),
        ("我又改名了，现在叫王五", "好的，王五！"),
    ]
    for q, a in turns:
        result = mem.add(
            [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
            user_id=user_id,
        )
        events = [
            (r.get("event"), (r.get("memory") or "")[:60]) for r in result.get("results", [])
        ]
        print(f"  add {q!r} → {events}")

    for query in ("我叫什么名字", "用户叫什么"):
        hits = mem.search(query, user_id=user_id, limit=5)
        print(f"  search({query!r}) →")
        for r in hits.get("results", []):
            print(f"    [{r['score']:.3f}] {r['memory'][:80]}")

    all_mems = mem.get_all(user_id=user_id)
    print(f"  get_all → {len(all_mems.get('results', []))} 条:")
    for r in all_mems.get("results", []):
        print(f"    - {r['memory'][:80]}")

    shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    client = QdrantClient(host=get_settings().qdrant_host, port=get_settings().qdrant_port)
    _clean_collections(client)

    _check_vector_consistency()
    _rename_scenario("默认 prompt", f"{SPIKE_PREFIX}_default", None)
    _rename_scenario("中文抽取 prompt", f"{SPIKE_PREFIX}_zh", FACT_EXTRACTION_PROMPT_ZH)

    _clean_collections(client)
    print("\nspike 完成")


if __name__ == "__main__":
    main()
