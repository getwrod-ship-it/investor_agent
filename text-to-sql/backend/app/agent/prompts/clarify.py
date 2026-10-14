"""
app/agent/prompts/clarify.py

Prompt templates for the CLARIFY node's two LLM paths (DESIGN_clarify.md):
  - AMBIGUOUS  : the guard flagged the question as unclear — generate one
                 targeted follow-up question for the user.
  - DIAGNOSIS  : retries were exhausted — explain the failure cause from the
                 real correction history and suggest how to rephrase.
The third path (infrastructure failure) never calls the LLM: the model
service itself is likely down, so it uses the static message in nodes/clarify.py.
"""

from app.agent.state import ChatTurn, CorrectionRecord
from app.agent.prompts.history import render_chat_history

# Truncation budgets — keep the diagnosis prompt inside a few hundred tokens
_ATTEMPT_SQL_MAX_CHARS = 500
_SCHEMA_CHUNK_MAX_CHARS = 800
_SCHEMA_CHUNKS = 2

CLARIFY_AMBIGUOUS_SYSTEM_PROMPT = """\
你是 Text-to-SQL 系统的追问助手。系统刚判定用户的问题缺少关键信息、无法生成有意义的 \
查询，你需要向用户提出一条追问。

规则：
- 只问缺失的信息，一次只问最关键的一维（查询对象 / 统计口径 / 范围），可以给 2 个候选 \
选项，例如："想按哪个口径看'最好卖'——按销售额还是按销量？"
- 禁止尝试回答原问题；禁止重复问最近对话中用户已给出的信息。
- 中文，不超过 120 字，直接输出追问文本，不要任何前后缀。

## 最近对话（同一会话，时间正序，可能为空）
{history}
"""

CLARIFY_AMBIGUOUS_USER_PROMPT = """\
用户最新问题：{question}
"""

CLARIFY_DIAGNOSIS_SYSTEM_PROMPT = """\
你是 Text-to-SQL 系统的失败诊断助手。系统对一个用户问题进行了多次 SQL 生成与自我修正，\
全部失败。请基于下面的真实记录，向用户解释失败原因并给出改问建议。

规则：
- 只依据 attempts 中的 error 原文和 schema 原文作判断，禁止编造不存在的表名、列名或原因。
- 输出中文，不超过 300 字，结构：
  ① 一句话结论（失败的核心原因）
  ② 多次尝试的共性错误点（如涉及，引用 error 关键词）
  ③ 2-3 条具体的改问建议
"""

CLARIFY_DIAGNOSIS_USER_PROMPT = """\
## 用户问题
{question}

## 失败的尝试记录
{attempts}

## 最后一次错误
{last_error}

## 可用的数据库表结构（节选）
{schema_context}
"""


def render_attempts(correction_history: list[CorrectionRecord]) -> str:
    """
    Render correction_history as the numbered attempts block of the diagnosis
    prompt: attempt number, the failed SQL (truncated), and its error.
    """
    if not correction_history:
        return "（无尝试记录）"
    blocks = []
    for record in correction_history:
        failed_sql = (record.get("failed_sql") or "").strip() or "（无 SQL）"
        if len(failed_sql) > _ATTEMPT_SQL_MAX_CHARS:
            failed_sql = failed_sql[:_ATTEMPT_SQL_MAX_CHARS] + "…"
        blocks.append(
            f"# 尝试 {record.get('attempt_number', '?')}\n"
            f"SQL：{failed_sql}\n"
            f"错误：{record.get('error_message', '')}"
        )
    return "\n\n".join(blocks)


def render_schema_context(retrieved_schema: list[str]) -> str:
    """
    Render the top schema chunks for the diagnosis prompt so failure analysis
    can reference tables that actually exist.
    """
    chunks = retrieved_schema[:_SCHEMA_CHUNKS]
    if not chunks:
        return "（未检索到表结构）"
    trimmed = []
    for chunk in chunks:
        if len(chunk) > _SCHEMA_CHUNK_MAX_CHARS:
            chunk = chunk[:_SCHEMA_CHUNK_MAX_CHARS] + "…"
        trimmed.append(chunk)
    return "\n\n".join(trimmed)


def render_ambiguous_history(chat_history: list[ChatTurn]) -> str:
    """Session memory block for the follow-up question prompt."""
    return render_chat_history(chat_history)
