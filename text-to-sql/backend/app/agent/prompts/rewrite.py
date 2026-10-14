"""
app/agent/prompts/rewrite.py

System and user prompt templates for the REWRITE node — rebuilds an
elliptical follow-up into one compact, self-contained retrieval question
using the K most recent session turns.
"""

REWRITE_SYSTEM_PROMPT = """\
你是检索问题改写器。用户在同一会话中连续提问，最新问题可能省略了主体、时间或指标，\
需要依赖上文才能理解。你的任务：结合最近对话，把它重构为一个独立完整、自包含的问题，\
供向量检索使用。

## 改写规则
- 只做指代消解和时间、主体、指标补全；不要改变用户的查询意图，不要新增需求。
- 跨轮指代（如"第一个呢"指向更早的轮次）要锚定对真正指代的那一轮。
- 相对时间要落成具体区间（如"前个月"→ 具体年月），以今天 {today} 为基准推断。
- 如果问题已经自包含、无需上文即可理解，原样返回并置 changed=false。

## 最近对话（同一会话，时间正序）
{history}

## Output format
Respond with a JSON object matching this exact schema:
{{
  "rewritten_question": "<改写后的完整问题>",
  "changed": <true | false>
}}
"""

REWRITE_USER_PROMPT = """\
Latest question: {question}
"""
