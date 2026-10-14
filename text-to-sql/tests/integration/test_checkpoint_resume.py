"""
Checkpoint-backed memory persistence — the core acceptance scenario.

Uses the REAL compiled graph topology and the REAL SqliteSaver (temp
checkpoints.db wired by conftest), with stub nodes that record the state
each node received. Verifies:
  1. memory survives rebuilding the graph instance (process restart)
  2. the rolling summary kicks in once the window overflows and also
     survives a rebuild
"""

import uuid
from types import SimpleNamespace

import app.agent.graph as graph_module
import app.services.conversation as conversation
import pytest
from app.agent.graph import build_graph
from app.services.conversation import append_turn

_STUB_NODES = (
    "reject", "rewrite", "retrieve", "generate",
    "validate", "execute", "correct", "clarify", "explain",
)


@pytest.fixture
def captured_states(monkeypatch):
    """Compile the real topology with recording stub nodes.

    The smalltalk path (guard → chat → END) avoids every LLM/RAG/MySQL
    dependency while exercising real checkpointing and routing.
    """
    captured: list[dict] = []

    def make_stub(result_fn):
        def stub(state):
            captured.append(dict(state))
            return {**state, **result_fn(state)}

        return stub

    monkeypatch.setattr(
        graph_module, "guard", make_stub(lambda s: {"intent": "smalltalk"})
    )
    monkeypatch.setattr(
        graph_module,
        "chat",
        make_stub(lambda s: {"direct_answer": "好的", "final_explanation": "好的"}),
    )
    for name in _STUB_NODES:
        monkeypatch.setattr(graph_module, name, make_stub(lambda s: {}))
    return captured


def _invoke_turn(graph, session: str, question: str) -> dict:
    final = graph.invoke(
        {"question": question, "session_id": session, "intent": "smalltalk"},
        {"configurable": {"thread_id": session}},
    )
    append_turn(graph, {"configurable": {"thread_id": session}}, final)
    return final


def _guard_states_for(captured: list[dict], question: str) -> list[dict]:
    # guard runs first for every turn; filter by the turn's question
    return [s for s in captured if s.get("question") == question]


def test_memory_survives_graph_rebuild(captured_states):
    session = f"sess-resume-{uuid.uuid4()}"

    graph1 = build_graph()
    _invoke_turn(graph1, session, "你好呀")
    captured_states.clear()  # only observe what the restarted process sees

    # simulate a process restart: brand-new compiled instance, same saver file
    graph2 = build_graph()
    _invoke_turn(graph2, session, "今天天气如何")

    second_turn_states = _guard_states_for(captured_states, "今天天气如何")
    assert second_turn_states, "guard stub did not run for the second turn"
    history = second_turn_states[0].get("chat_history")
    assert [t["question"] for t in history] == ["你好呀"]
    assert history[0]["sql"] is None  # smalltalk turn carries no SQL
    assert history[0]["answer"] == "好的"


def test_summary_folds_after_window_overflow_and_survives_rebuild(
    captured_states, monkeypatch
):
    class FakeSummaryLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content="用户在查 litemall_order 的销量口径")

    monkeypatch.setattr(
        conversation, "get_chat_llm", lambda temperature=0: FakeSummaryLLM()
    )

    session = f"sess-overflow-{uuid.uuid4()}"
    graph = build_graph()
    for i in range(1, 7):  # k = 5: the 6th turn must trigger the fold
        _invoke_turn(graph, session, f"问题{i}")

    snap = graph.get_state({"configurable": {"thread_id": session}})
    assert len(snap.values["chat_history"]) == 5
    assert snap.values["history_summary"] == "用户在查 litemall_order 的销量口径"
    assert snap.values["memory_turn_count"] == 6

    # after a restart the summary reaches the nodes again
    captured_states.clear()
    graph2 = build_graph()
    _invoke_turn(graph2, session, "问题7")
    guard_state = _guard_states_for(captured_states, "问题7")[0]
    assert guard_state["history_summary"] == "用户在查 litemall_order 的销量口径"
    assert len(guard_state["chat_history"]) == 5
