"""
app/config.py

Pydantic-settings BaseSettings — all configuration is read from the .env file
at the project root.

The two database roles are strictly separated:
  - business_database_url : the USER's business MySQL database (read-only
    access), e.g. mysql+pymysql://user:pass@host:3306/dbname
  - system_db_path        : a private SQLite file where the agent stores its own
    bookkeeping tables (query history, correction attempts, schema snapshots).

Usage:
    from app.config import get_settings
    settings = get_settings()
"""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from the project root regardless of working directory
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from the root .env file."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",  # ignore Docker-injected vars and other extras
    )

    # DeepSeek (SQL generation LLM). NOTE: "deepseek-v4-flash" is a legacy id —
    # the current Flash model is served as "deepseek-flash" (DeepSeek-V4.1-Flash).
    deepseek_api_key: SecretStr
    deepseek_model: str = "deepseek-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    # Client-level resilience (DESIGN_clarify.md DR-7): timeout must stay
    # generous — the thinking model is slow; retries absorb network blips.
    deepseek_timeout_seconds: int = 60
    deepseek_max_retries: int = 2

    # Local embedding / reranker models, loaded from disk
    # (see bge-embed-rerank-deploy.md for download and GPU setup)
    embedding_model: str = "models/bge-m3"
    embedding_device: str = "cuda"
    reranker_model: str = "models/bge-reranker-v2-m3"

    # Business database — the user's own MySQL, accessed read-only
    business_database_url: SecretStr
    business_db_exclude_tables: str = ""  # comma-separated table names to skip

    # Agent system database — private SQLite file for internal bookkeeping
    system_db_path: str = "data/system.db"

    # LangGraph checkpoint — conversation state persistence (thread_id = session_id)
    checkpoint_db_path: str = "data/checkpoints.db"

    # Memory compression — rolling summary of turns that fell out of the window
    memory_summary_enabled: bool = True
    memory_summary_max_chars: int = 600

    # Qdrant (vector store). The backend runs on the host (GPU local models),
    # so the default points at a Qdrant container with a published port.
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Agent behaviour
    max_retries: int = 3
    query_timeout_seconds: int = 30
    query_row_limit: int = 1000
    # GUARD ambiguous → CLARIFY routing switch (kill-switch for misjudged
    # follow-up questions; guard keeps classifying, only the route changes)
    clarify_ambiguity_enabled: bool = True
    # EXPLAIN answerability gate + semantic-failure retry loop (kill-switch for
    # misjudged "cannot answer" verdicts; off = legacy free-text EXPLAIN, the
    # graph is topologically unchanged and the loop never fires)
    explain_answerability_enabled: bool = True

    # Session short-term memory — how many recent turns (same session_id)
    # are injected into GUARD / REWRITE / GENERATE context
    memory_window_size: int = 5

    # Long-term memory (mem0, see DESIGN_long_term_memory.md) — cross-session
    # facts + episodic recall. master switch off → recall returns empty and
    # write_turn is a no-op (fail-open, tests run with it off).
    memory_enabled: bool = True
    mem0_user_id: str = "default"  # single-user identity key (DR-L3)
    mem0_facts_collection: str = "text2sql_memory_facts"
    mem0_episodic_collection: str = "text2sql_memory_episodic"
    mem0_facts_history_db: str = "data/mem0_facts.db"
    mem0_episodic_history_db: str = "data/mem0_episodic.db"
    mem0_facts_top_k: int = 5
    mem0_facts_score_threshold: float = 0.5
    mem0_episodic_top_k: int = 2
    mem0_recall_timeout_seconds: float = 1.5

    # Local decision model (StartLux-Decision /v1/systemone, see
    # docs/DESIGN_local_decision_model.md). Master switch defaults OFF — with
    # it off (or the service down) every decide_* gate returns None and
    # behaviour is byte-identical to the DeepSeek-only pipeline (DR-2).
    decision_enabled: bool = False
    decision_base_url: str = "http://127.0.0.1:9090"
    # steady state ~100ms/question; 2s covers cold start / GPU contention
    decision_timeout_seconds: float = 2.0
    # log-only mode: local result is recorded, DeepSeek still answers (DR-3)
    decision_shadow_mode: bool = False
    decision_guard_enabled: bool = True          # ① GUARD cascade
    decision_intent_threshold: float = 0.70      # choice top-1 adoption threshold
    decision_ltm_gate_enabled: bool = True       # ② long-term recall gate
    decision_ltm_gate_threshold: float = 0.60    # skip recall only when P(need) < this
    decision_write_gate_enabled: bool = True     # ③ facts-write gate
    decision_rewrite_gate_enabled: bool = True   # ④ rewrite gate

    # Application
    app_env: str = "local"

    @field_validator("business_database_url")
    @classmethod
    def _validate_mysql_url(cls, value: SecretStr) -> SecretStr:
        """Only MySQL connection strings are supported."""
        scheme = value.get_secret_value().split(":", 1)[0].split("+", 1)[0].lower()
        if scheme != "mysql":
            raise ValueError(
                "BUSINESS_DATABASE_URL must be a MySQL connection string, e.g. "
                f"mysql+pymysql://user:password@host:3306/dbname (got scheme '{scheme}')"
            )
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def system_database_url(self) -> str:
        """SQLAlchemy URL for the internal SQLite system database."""
        return f"sqlite:///{self.system_db_path}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def business_db_dialect(self) -> str:
        """Human-readable dialect name — the agent only supports MySQL."""
        return "MySQL"

    @property
    def excluded_tables(self) -> frozenset[str]:
        """Parse business_db_exclude_tables into a set of table names."""
        return frozenset(
            name.strip() for name in self.business_db_exclude_tables.split(",") if name.strip()
        )


def load_settings() -> Settings:
    """Load settings from the .env file."""
    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return load_settings()
