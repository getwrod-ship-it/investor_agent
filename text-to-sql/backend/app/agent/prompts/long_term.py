"""
app/agent/prompts/long_term.py

Shared renderer that turns the long-term memory recall (facts + episodic
excerpts) into a compact prompt block. GENERATE and CHAT consume the same
rendering. Facts are marked as the current-valid source of truth — on any
conflict with episodic excerpts or the session history, facts win (DR-L5 /
DESIGN_long_term_memory.md §4.5).
"""

_FACTS_HEADER = "## 长期记忆·当前有效的事实与口径（跨会话）"
_EPISODES_HEADER = "## 相关历史片段（过往会话节选，供参考）"
_CONFLICT_NOTE = "（若与下方历史片段或本会话对话冲突，以本块为准——它是最新确认的现值）"
_EMPTY = "（无）"


def render_long_term_memory(facts: list[str], episodes: list[str]) -> str:
    """
    Render the long-term memory prompt block.

    Both legs empty → empty string (the block is omitted entirely). When at
    least one leg has content, both sub-blocks are rendered so the model sees
    both categories deterministically; an empty leg shows the "（无）"
    placeholder, mirroring render_chat_history's semantics.

    Args:
        facts:    Already-formatted current-valid fact lines (may be empty).
        episodes: Already-formatted episodic excerpt lines (may be empty).
    """
    if not facts and not episodes:
        return ""

    facts_block = _EMPTY
    if facts:
        fact_lines = "\n".join(f"- {fact}" for fact in facts)
        facts_block = f"{fact_lines}\n{_CONFLICT_NOTE}"

    episodes_block = _EMPTY
    if episodes:
        episodes_block = "\n".join(f"- {episode}" for episode in episodes)

    return f"{_FACTS_HEADER}\n{facts_block}\n\n{_EPISODES_HEADER}\n{episodes_block}"
