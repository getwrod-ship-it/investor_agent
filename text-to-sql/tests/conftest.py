"""
tests/conftest.py

Test-wide environment setup and shared fixtures.

The env setup at the top MUST run before any `app.*` import:
- fake credentials so Settings() validates without touching real services
  (real env vars override .env, so tests never use the real API key)
- SYSTEM_DB_PATH pointed at a throwaway temp dir: importing app.db creates
  the engine at import time, and without this it would touch the real
  data/system.db
- CHECKPOINT_DB_PATH likewise: importing app.agent.graph builds the graph
  with the checkpointer singleton at import time
"""

import os
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path

_TEST_ENV_DIR = Path(tempfile.mkdtemp(prefix="text2sql-tests-"))

os.environ["DEEPSEEK_API_KEY"] = "sk-test-fake-key"
os.environ["BUSINESS_DATABASE_URL"] = "mysql+pymysql://user:pass@localhost:3306/testdb"
os.environ["SYSTEM_DB_PATH"] = str(_TEST_ENV_DIR / "system.db")
os.environ["CHECKPOINT_DB_PATH"] = str(_TEST_ENV_DIR / "checkpoints.db")
# Long-term memory off: recall is empty / writes are no-ops, and mem0 is
# never even imported — the existing test surface is untouched.
os.environ["MEMORY_ENABLED"] = "false"
# Local decision gates off: every decide_* returns None (no HTTP attempted)
# and the pipeline is byte-identical to the DeepSeek-only path (fail-open).
os.environ["DECISION_ENABLED"] = "false"
os.environ["APP_ENV"] = "test"

import pytest  # noqa: E402
from app.agent.state import AgentState  # noqa: E402
from app.config import get_settings  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine  # noqa: E402

# Re-read settings with the test env (defensive: in case anything imported
# app.config earlier) and register tables on the shared SQLModel metadata.
get_settings.cache_clear()
import app.models.db  # noqa: F401,E402


def _base_state() -> AgentState:
    """A complete AgentState dict — covers every field, new ones included."""
    return {
        # Input
        "question": "上个月销售额最高的三个商品是什么？",
        "session_id": "sess-test",
        # Intent guard
        "intent": "data_query",
        "direct_answer": None,
        "rewritten_question": None,
        # Session memory
        "chat_history": [],
        "history_summary": None,
        "memory_turn_count": 0,
        # Long-term memory (route-layer recall, reset per turn like the rest)
        "long_term_facts": [],
        "episodic_memories": [],
        # RAG context
        "retrieved_schema": [],
        "retrieved_docs": [],
        # SQL generation
        "generated_sql": None,
        "sql_reasoning": None,
        "sql_assumptions": [],
        "sql_confidence": 0.0,
        # Validation / execution
        "validation_error": None,
        "execution_result": None,
        "execution_error": None,
        # Self-correction
        "retry_count": 0,
        "correction_history": [],
        # Clarification / final output
        "needs_clarification": False,
        "clarification_message": None,
        "clarify_reason": None,
        "final_explanation": None,
        "answer_failure_reason": None,
        # Observability
        "trace_id": str(uuid.uuid4()),
    }


@pytest.fixture
def make_state():
    """Factory for a full AgentState dict; overrides replace matching keys."""

    def _factory(**overrides) -> AgentState:
        state = _base_state()
        state.update(overrides)
        return state

    return _factory


@pytest.fixture
def db_session(tmp_path: Path) -> Generator[Session, None, None]:
    """A SQLModel session on a fresh temporary SQLite database."""
    engine = create_engine(
        f"sqlite:///{tmp_path.as_posix()}/sys.db",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()
