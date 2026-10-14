"""
render_chat_history summary extension: with a summary the block gains a
"更早对话摘要" header; without one the output stays byte-identical to the
pre-compression renderer (regression protection).
"""

from app.agent.prompts.history import render_chat_history

_HISTORY = [
    {
        "question": "上个月销售额最高的三个商品是什么？",
        "sql": "SELECT goods_name FROM litemall_order",
        "answer": "TOP3 是 A、B、C",
    },
    {
        "question": "那前个月呢？",
        "sql": None,
        "answer": "TOP3 是 D、E、F",
    },
]


def test_without_summary_output_has_no_headers():
    block = render_chat_history(_HISTORY)  # type: ignore[arg-type]
    assert "##" not in block
    assert block.startswith("1. 问：上个月销售额最高的三个商品是什么？")
    assert "（无 SQL）" in block  # smalltalk-style turn keeps its placeholder


def test_without_summary_explicit_none_matches_default():
    assert render_chat_history(_HISTORY, summary=None) == render_chat_history(  # type: ignore[arg-type]
        _HISTORY  # type: ignore[arg-type]
    )


def test_with_summary_renders_summary_then_recent_block():
    block = render_chat_history(_HISTORY, summary="早期聊过 litemall_order 的销售额口径", empty_text="（无）")  # type: ignore[arg-type]

    assert block.startswith("## 更早对话摘要\n早期聊过 litemall_order 的销售额口径")
    assert "## 最近对话\n1. 问：上个月销售额最高的三个商品是什么？" in block
    assert "2. 问：那前个月呢？" in block


def test_summary_only_without_window_turns():
    block = render_chat_history([], summary="只有摘要没有窗口轮", empty_text="（无历史对话）")
    assert block == "## 更早对话摘要\n只有摘要没有窗口轮"


def test_empty_history_and_no_summary_render_placeholder():
    assert render_chat_history([], summary=None) == "（无历史对话）"
