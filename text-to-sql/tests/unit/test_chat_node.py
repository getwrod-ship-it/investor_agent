"""
CHAT node: smalltalk answered directly with the data-assistant persona.
"""

from app.agent.nodes.chat import _FALLBACK_ANSWER, chat
from langchain_core.messages import AIMessage


class FakeChatLLM:
    def __init__(self, content: str = "", error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return AIMessage(content=self.content)


def _patch_llm(monkeypatch, fake: FakeChatLLM):
    captured = {}

    def factory(temperature: float = 0):
        captured["temperature"] = temperature
        return fake

    monkeypatch.setattr("app.agent.nodes.chat.get_chat_llm", factory)
    return captured


def test_chat_writes_direct_answer_and_final_explanation(make_state, monkeypatch):
    _patch_llm(monkeypatch, FakeChatLLM(content="我能帮你查订单、商品等数据。"))

    result = chat(make_state(question="你能帮我做什么？", intent="smalltalk"))

    assert result["direct_answer"] == "我能帮你查订单、商品等数据。"
    # mirrored into the audit field so it persists on the QueryRun record
    assert result["final_explanation"] == result["direct_answer"]
    assert result["needs_clarification"] is False


def test_chat_uses_persona_temperature(make_state, monkeypatch):
    captured = _patch_llm(monkeypatch, FakeChatLLM(content="ok"))

    chat(make_state(question="你好"))

    assert captured["temperature"] == 0.7


def test_chat_falls_back_on_llm_error(make_state, monkeypatch):
    _patch_llm(monkeypatch, FakeChatLLM(error=RuntimeError("api down")))

    result = chat(make_state(question="你好"))

    assert result["direct_answer"] == _FALLBACK_ANSWER
    assert "litemall" in result["direct_answer"]
