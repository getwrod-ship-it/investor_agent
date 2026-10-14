"""
app/agent/prompts/summary.py

Prompt for the rolling memory summary: folds the turns that just fell out
of the detailed window (plus the previous summary) into one updated digest.
Domain-tuned: table names, metric definitions, numeric conclusions and time
ranges must survive the compression — they are what late-session follow-ups
resolve against.
"""

SUMMARY_SYSTEM_PROMPT = """你是数据库查询助手的记忆压缩器。请把【已有摘要】与【新结束的对话轮】合并为一份更新后的摘要，供后续多轮对话理解上下文。
必须保留：涉及的业务表与字段、指标口径、数值结论（保留数字）、时间范围、用户意图的演变、未结束的追问。
必须丢弃：寒暄、SQL 语法细节、报错信息、与查数无关的内容。
已有摘要（可能为空）：{old_summary}
新结束的对话轮：
{fallen_turns}
要求：不超过 {max_chars} 字；直接输出摘要正文，不要任何前缀、标题或解释。"""
