"""
app/business_db.py

SQLAlchemy engine for the USER's business MySQL database (configured via
BUSINESS_DATABASE_URL). This connection is used read-only:

  - app/rag/crawler.py          crawls table/column/FK metadata for the vector store
  - app/tools/mysql_executor.py executes agent-generated SELECT queries

NullPool is used deliberately: each execution gets a fresh connection, so no
session state (e.g. MySQL max_execution_time) leaks between queries and stale
connections never survive a business database restart or swap.
"""

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from app.config import get_settings


@lru_cache(maxsize=1)
def get_business_engine() -> Engine:
    """Create and return the SQLAlchemy engine for the configured business database."""
    settings = get_settings()
    url = settings.business_database_url.get_secret_value()
    return create_engine(url, poolclass=NullPool, pool_pre_ping=True)
