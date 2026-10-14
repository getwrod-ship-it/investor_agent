"""
GENERATE node: session history injection into the system prompt.
"""

from app.agent.nodes.generate import _format_history_block


def test_empty_history_renders_placeholder():
    assert _format_history_block([]) == "（本轮是会话中的第一个问题）"


def test_history_renders_numbered_turns():
    history = [
        {
            "question": "上个月销售额最高的三个商品是什么？",
            "sql": "SELECT goods_name FROM t",
            "answer": "TOP3 是A/B/C",
        },
        {
            "question": "你好",
            "sql": None,
            "answer": "我是查询助手",
        },
    ]
    block = _format_history_block(history)  # type: ignore[arg-type]

    lines = block.splitlines()
    assert lines[0].startswith("1. 问：上个月销售额最高的三个商品是什么？")
    assert "SQL：SELECT goods_name FROM t" in lines[1]
    assert "答：TOP3 是A/B/C" in lines[2]
    # smalltalk turns have no SQL — the placeholder must appear
    assert any("（无 SQL）" in line for line in lines)


def test_generate_prompt_contains_history_block(make_state, monkeypatch):
    from app.agent.nodes.generate import SQLGenerationResponse, generate
    from langchain_core.messages import AIMessage

    captured: dict = {}

    class FakeStructuredLLM:
        def invoke(self, messages):
            captured["messages"] = messages
            return {
                "parsed": SQLGenerationResponse(
                    sql="SELECT 1", reasoning="r", assumptions=[], confidence=0.9
                ),
                "raw": AIMessage(content="{}"),
            }

    monkeypatch.setattr(
        "app.agent.nodes.generate.get_structured_llm",
        lambda schema: FakeStructuredLLM(),
    )

    history = [
        {
            "question": "上个月销售额最高的三个商品是什么？",
            "sql": "SELECT goods_name FROM litemall_order",
            "answer": "TOP3 是A/B/C",
        }
    ]
    generate(
        make_state(
            question="那前个月呢？",
            retrieved_schema=["CREATE TABLE litemall_order (...)"],
            retrieved_docs=["有效订单 = order_status IN (201,301,401)"],
            chat_history=history,  # type: ignore[arg-type]
        )
    )

    system_content = captured["messages"][0]["content"]
    assert "1. 问：上个月销售额最高的三个商品是什么？" in system_content
    assert "SELECT goods_name FROM litemall_order" in system_content
    # generate keeps the ORIGINAL question (rewrite is retrieval-only)
    user_content = captured["messages"][1]["content"]
    assert "那前个月呢？" in user_content
