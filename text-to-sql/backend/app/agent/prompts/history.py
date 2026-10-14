"""
app/agent/prompts/history.py

Shared renderer that turns the session memory into a compact prompt block:
the rolling summary of older turns (if any) plus the recent turns as a
numbered list. GUARD, REWRITE and GENERATE all consume the same rendering
so every node sees an identical view of the session.
"""

from typing import Optional

from app.agent.state import ChatTurn


def _render_turns(chat_history: list[ChatTurn]) -> str:
    """Numbered 问 / SQL / 答 lines, one block per turn."""
    lines = []
    for index, turn in enumerate(chat_history, start=1):
        sql = turn.get("sql") or "（无 SQL）"
        answer = turn.get("answer") or ""
        lines.append(
            f"{index}. 问：{turn.get('question', '')}\n   SQL：{sql}\n   答：{answer}"
        )
    return "\n".join(lines)


def render_chat_history(
    chat_history: list[ChatTurn],
    empty_text: str = "（无历史对话）",
    summary: Optional[str] = None,
) -> str:
    """
    Render session memory as a prompt block.

    Without a summary the output is the plain numbered list — byte-identical
    to the pre-compression renderer, so sessions without compression keep
    their exact prompt shape. With one, the summary heads the block and the
    recent turns follow under a "最近对话" header.

    Args:
        chat_history: Turns in chronological order; may be empty.
        empty_text:   Placeholder rendered when there is nothing at all.
        summary:      Rolling summary of turns older than the window.
    """
    if not summary:
        if not chat_history:
            return empty_text
        return _render_turns(chat_history)
    parts = [f"## 更早对话摘要\n{summary}"]
    if chat_history:
        parts.append(f"## 最近对话\n{_render_turns(chat_history)}")
    return "\n\n".join(parts)
