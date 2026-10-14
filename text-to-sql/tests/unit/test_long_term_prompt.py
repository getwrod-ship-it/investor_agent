"""
Unit tests for the long-term memory prompt rendering — the shared block
renderer plus its consumption by the GENERATE / CHAT prompt templates.
"""

from app.agent.nodes.generate import _format_long_term_block
from app.agent.prompts.chat import CHAT_SYSTEM_PROMPT
from app.agent.prompts.generate import GENERATE_SYSTEM_PROMPT
from app.agent.prompts.long_term import render_long_term_memory

FACTS = ["名叫王五", "销售额按含税口径计算"]
EPISODES = ["[2026-10-12｜数据查询｜会话a3f2c1de] 问：退款TOP3？｜答：TOP1 是沙发"]


def test_both_empty_returns_empty_string():
    assert render_long_term_memory([], []) == ""


def test_facts_only_shows_placeholder_for_episodes():
    block = render_long_term_memory(FACTS, [])
    assert "长期记忆·当前有效的事实与口径" in block
    assert "- 名叫王五" in block and "- 销售额按含税口径计算" in block
    assert "以本块为准" in block  # conflict rule: facts win
    assert "相关历史片段" in block
    assert block.endswith("（无）")


def test_episodes_only_shows_placeholder_for_facts():
    block = render_long_term_memory([], EPISODES)
    assert "当前有效的事实与口径" in block
    assert "（无）" in block
    assert "- [2026-10-12｜数据查询" in block


def test_both_present_renders_two_blocks():
    block = render_long_term_memory(FACTS, EPISODES)
    assert block.index("当前有效的事实与口径") < block.index("相关历史片段")
    assert block.count("（无）") == 0
    assert "- 名叫王五" in block and "- [2026-10-12" in block


# --------------------------------------------- consumption by node prompts


def test_generate_block_from_state(make_state):
    state = make_state(long_term_facts=FACTS, episodic_memories=EPISODES)
    block = _format_long_term_block(state)
    assert "- 名叫王五" in block

    # and the GENERATE template accepts the rendered block
    prompt = GENERATE_SYSTEM_PROMPT.format(
        schema_context="s", docs_context="d", history="h",
        long_term_memory=block, dialect="MySQL",
    )
    assert "长期记忆·当前有效的事实与口径" in prompt
    assert prompt.index("## 最近对话") < prompt.index("长期记忆·当前有效的事实与口径")
    assert prompt.index("长期记忆·当前有效的事实与口径") < prompt.index("## SQL conventions")


def test_generate_block_empty_state_renders_nothing(make_state):
    assert _format_long_term_block(make_state()) == ""
    # and the template still formats cleanly with an empty block
    prompt = GENERATE_SYSTEM_PROMPT.format(
        schema_context="s", docs_context="d", history="h",
        long_term_memory="", dialect="MySQL",
    )
    assert "长期记忆" not in prompt


def test_chat_prompt_carries_long_term_block():
    block = render_long_term_memory(FACTS, [])
    system = CHAT_SYSTEM_PROMPT.format(long_term_memory=block)
    assert "名叫王五" in system
    assert "litemall" in system  # original persona intact


def test_chat_prompt_without_memory_stays_clean():
    system = CHAT_SYSTEM_PROMPT.format(long_term_memory="")
    assert "长期记忆" not in system
    assert "litemall" in system
