"""
Sessions API integration tests: list order, chronological replay
(including unsafe refusals), 404s, and DELETE with cascade + checkpoint
thread cleanup. Seeded directly via SQLModel — no /query round-trip needed.
"""

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import app.api.routes.sessions as sessions_route
import pytest
from app.db import get_session
from app.main import app
from app.models.db import ChatSession, CorrectionAttempt, QueryRun
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, select

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def temp_engine(tmp_path: Path):
    engine = create_engine(
        f"sqlite:///{tmp_path.as_posix()}/sys.db",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def client(temp_engine) -> Generator[TestClient, None, None]:
    def override_get_session():
        with Session(temp_engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_list_sessions_newest_first(client, temp_engine):
    with Session(temp_engine) as db:
        for sid, minutes in (("older", 1), ("newer", 2)):
            db.add(
                ChatSession(
                    id=sid,
                    title=sid,
                    created_at=_BASE,
                    updated_at=_BASE + timedelta(minutes=minutes),
                )
            )
        db.commit()

    resp = client.get("/sessions")

    assert resp.status_code == 200
    data = resp.json()
    assert [s["session_id"] for s in data] == ["newer", "older"]
    assert set(data[0].keys()) == {
        "session_id", "title", "turn_count", "last_question",
        "created_at", "updated_at",
    }


def test_detail_replays_chronologically_including_unsafe(client, temp_engine):
    with Session(temp_engine) as db:
        db.add(ChatSession(id="s1", title="会话一", turn_count=3))
        db.add(
            QueryRun(
                session_id="s1",
                question="问题一",
                intent="data_query",
                generated_sql="SELECT 1",
                execution_result=[{"a": 1}],
                final_explanation="答一",
                created_at=_BASE,
            )
        )
        db.add(
            QueryRun(
                session_id="s1",
                question="闲聊",
                intent="smalltalk",
                final_explanation="我是助手",
                created_at=_BASE + timedelta(minutes=1),
            )
        )
        db.add(
            QueryRun(
                session_id="s1",
                question="危险",
                intent="unsafe",
                created_at=_BASE + timedelta(minutes=2),
            )
        )
        db.commit()

    resp = client.get("/sessions/s1")

    assert resp.status_code == 200
    msgs = resp.json()["messages"]
    assert [m["question"] for m in msgs] == ["问题一", "闲聊", "危险"]
    first = msgs[0]
    assert first["sql"] == "SELECT 1"
    assert first["result"] == [{"a": 1}]
    assert first["row_count"] == 1
    assert first["succeeded"] is True
    assert msgs[1]["direct_answer"] == "我是助手"  # replay of a smalltalk reply
    assert msgs[1]["succeeded"] is True
    assert msgs[2]["succeeded"] is False  # refusal counts as failure
    assert msgs[2]["direct_answer"] is None


def test_detail_unknown_session_404(client):
    assert client.get("/sessions/nope").status_code == 404


def test_delete_unknown_session_404(client):
    assert client.delete("/sessions/nope").status_code == 404


def test_delete_cascades_and_clears_checkpoint_thread(
    client, temp_engine, monkeypatch
):
    with Session(temp_engine) as db:
        run = QueryRun(session_id="s9", question="q", intent="data_query", created_at=_BASE)
        db.add(run)
        db.commit()
        db.add(
            CorrectionAttempt(
                query_run_id=run.id,
                attempt_number=1,
                failed_sql="SELECT 1",
                error_message="err",
            )
        )
        db.add(ChatSession(id="s9", title="t"))
        db.commit()

    deleted: list[str] = []

    class FakeSaver:
        def delete_thread(self, thread_id: str):
            deleted.append(thread_id)

    monkeypatch.setattr(sessions_route, "get_checkpointer", lambda: FakeSaver())

    resp = client.delete("/sessions/s9")

    assert resp.status_code == 204
    assert deleted == ["s9"]
    with Session(temp_engine) as db:
        assert db.exec(select(QueryRun)).all() == []
        assert db.exec(select(CorrectionAttempt)).all() == []
        assert db.get(ChatSession, "s9") is None
