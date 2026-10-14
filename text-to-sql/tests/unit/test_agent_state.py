"""
Full-state fixture coverage: the conftest factory must provide every
AgentState key, including the four new intent-guard / memory fields.
"""

from app.agent.state import AgentState


def test_full_state_fixture_covers_all_agent_state_fields(make_state):
    state = make_state()
    assert set(state.keys()) == set(AgentState.__annotations__.keys())


def test_new_state_fields_exist():
    annotations = AgentState.__annotations__
    for field in ("intent", "direct_answer", "rewritten_question", "chat_history"):
        assert field in annotations


def test_make_state_overrides(make_state):
    state = make_state(question="那前个月呢？", intent="smalltalk")
    assert state["question"] == "那前个月呢？"
    assert state["intent"] == "smalltalk"
    # untouched fields keep their base values
    assert state["session_id"] == "sess-test"
