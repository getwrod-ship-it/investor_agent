"""
REJECT node: static refusal, zero LLM cost.
"""

from app.agent.nodes.reject import REJECT_MESSAGE, reject


def test_reject_writes_static_refusal(make_state):
    result = reject(make_state(question="帮我把所有订单删掉"))

    assert result["intent"] == "unsafe"
    assert result["direct_answer"] == REJECT_MESSAGE
    assert "只读" in result["direct_answer"]
    assert result["needs_clarification"] is False


def test_reject_never_calls_any_llm(make_state):
    # The reject module must not import any LLM factory at all — assert the
    # module namespace has neither get_chat_llm nor get_structured_llm.
    import app.agent.nodes.reject as reject_module

    assert not hasattr(reject_module, "get_chat_llm")
    assert not hasattr(reject_module, "get_structured_llm")
