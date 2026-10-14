"""
Guard-layer routing logic and graph wiring.
"""

from types import SimpleNamespace

from app.agent.graph import (
    build_graph,
    route_after_explain,
    route_after_guard,
    route_after_validate,
)
from langgraph.graph import END


def test_guard_routes_smalltalk_to_chat(make_state):
    assert route_after_guard(make_state(intent="smalltalk")) == "chat"


def test_guard_routes_unsafe_to_reject(make_state):
    assert route_after_guard(make_state(intent="unsafe")) == "reject"


def test_guard_routes_data_query_to_rewrite(make_state):
    assert route_after_guard(make_state(intent="data_query")) == "rewrite"


def test_guard_defaults_to_rewrite_when_intent_missing(make_state):
    state = {k: v for k, v in make_state().items() if k != "intent"}
    assert route_after_guard(state) == "rewrite"
    assert route_after_guard(make_state(intent=None)) == "rewrite"


def test_guard_routes_ambiguous_to_clarify(make_state):
    assert route_after_guard(make_state(intent="ambiguous")) == "clarify"


def test_guard_routes_ambiguous_to_rewrite_when_disabled(make_state, monkeypatch):
    monkeypatch.setattr(
        "app.agent.graph.get_settings",
        lambda: SimpleNamespace(clarify_ambiguity_enabled=False, max_retries=3),
    )
    assert route_after_guard(make_state(intent="ambiguous")) == "rewrite"


def test_validate_routes_to_execute_when_clean(make_state):
    assert route_after_validate(make_state(validation_error=None)) == "execute"


def test_validate_routes_to_correct_when_error_and_retries_left(make_state):
    state = make_state(validation_error="bad sql", retry_count=1)
    assert route_after_validate(state) == "correct"


def test_validate_routes_to_clarify_when_retries_exhausted(make_state, monkeypatch):
    monkeypatch.setattr(
        "app.agent.graph.get_settings",
        lambda: SimpleNamespace(clarify_ambiguity_enabled=True, max_retries=3),
    )
    state = make_state(validation_error="No SQL was generated", retry_count=3)
    assert route_after_validate(state) == "clarify"


def test_explain_routes_to_end_when_no_retry_requested(make_state):
    assert route_after_explain(make_state(answer_failure_reason=None)) == END


def test_explain_routes_to_correct_on_semantic_retry(make_state):
    state = make_state(
        answer_failure_reason="查到 0 行；名称匹配串过窄", retry_count=1
    )
    assert route_after_explain(state) == "correct"


def test_explain_routes_to_end_defensively_when_exhausted(make_state, monkeypatch):
    # unreachable in practice (final-attempt EXPLAIN clears the reason) —
    # the defensive branch guarantees no loop past the quota regardless
    monkeypatch.setattr(
        "app.agent.graph.get_settings",
        lambda: SimpleNamespace(clarify_ambiguity_enabled=True, max_retries=3),
    )
    state = make_state(answer_failure_reason="还想重试", retry_count=3)
    assert route_after_explain(state) == END


def test_graph_contains_guard_layer_nodes():
    compiled = build_graph()
    node_ids = set(compiled.get_graph().nodes.keys())
    expected = {
        "guard", "chat", "reject", "rewrite",
        "retrieve", "generate", "validate", "execute",
        "correct", "clarify", "explain",
    }
    assert expected <= node_ids


def test_graph_entry_point_is_guard():
    compiled = build_graph()
    edges = [(edge.source, edge.target) for edge in compiled.get_graph().edges]
    assert ("__start__", "guard") in edges


def test_guard_has_four_conditional_targets():
    compiled = build_graph()
    graph = compiled.get_graph()
    targets = {
        edge.target for edge in graph.edges if edge.source == "guard"
    }
    assert {"rewrite", "chat", "reject", "clarify"} <= targets


def test_validate_has_three_conditional_targets():
    compiled = build_graph()
    graph = compiled.get_graph()
    targets = {
        edge.target for edge in graph.edges if edge.source == "validate"
    }
    assert {"execute", "correct", "clarify"} <= targets


def test_explain_has_two_conditional_targets():
    # semantic-retry loop (DESIGN_explain_answerability.md): EXPLAIN ends the
    # run or loops back to CORRECT — no longer an unconditional terminal edge
    compiled = build_graph()
    graph = compiled.get_graph()
    targets = {
        edge.target for edge in graph.edges if edge.source == "explain"
    }
    assert {"correct", "__end__"} <= targets


def test_rewrite_feeds_retrieve():
    compiled = build_graph()
    edges = [(edge.source, edge.target) for edge in compiled.get_graph().edges]
    assert ("rewrite", "retrieve") in edges


def test_chat_and_reject_are_terminal():
    compiled = build_graph()
    graph = compiled.get_graph()
    for node in ("chat", "reject"):
        targets = {edge.target for edge in graph.edges if edge.source == node}
        assert targets == {"__end__"}
