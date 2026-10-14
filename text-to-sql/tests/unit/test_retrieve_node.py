"""
RETRIEVE node: retrieval must use the rewritten question when present.
"""

from app.agent.nodes.retrieve import retrieve
from app.rag.retriever import RetrievedChunk


def test_retrieve_uses_rewritten_question(make_state, monkeypatch):
    calls: list[str] = []

    def fake_retrieve_context(question: str, top_k: int = 3):
        calls.append(question)
        return [
            RetrievedChunk(text="-- table a", score=0.9, source="schema", metadata={}),
            RetrievedChunk(text="口径说明", score=0.8, source="docs", metadata={}),
        ]

    monkeypatch.setattr(
        "app.agent.nodes.retrieve.retrieve_context", fake_retrieve_context
    )

    result = retrieve(
        make_state(
            question="那前个月呢？",
            rewritten_question="2026年8月销售额最高的三个商品是什么？",
        )
    )

    assert calls == ["2026年8月销售额最高的三个商品是什么？"]
    assert result["retrieved_schema"] == ["-- table a"]
    assert result["retrieved_docs"] == ["口径说明"]


def test_retrieve_falls_back_to_original_question(make_state, monkeypatch):
    calls: list[str] = []

    def fake_retrieve_context(question: str, top_k: int = 3):
        calls.append(question)
        return []

    monkeypatch.setattr(
        "app.agent.nodes.retrieve.retrieve_context", fake_retrieve_context
    )

    retrieve(make_state(question="上个月销量如何？", rewritten_question=None))

    assert calls == ["上个月销量如何？"]
