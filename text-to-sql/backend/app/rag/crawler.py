"""
app/rag/crawler.py

Business database schema crawler.
crawl_schema() uses the SQLAlchemy inspector to read table metadata, column
definitions, and foreign key relationships from the MySQL business database,
returning one SchemaMetadata dataclass per business table.
"""

import logging
from dataclasses import dataclass, field

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Tables that are never business data (migration bookkeeping etc.)
_EXCLUDED_TABLES = frozenset({"alembic_version"})

# SQLite internal tables all share this prefix
_EXCLUDED_PREFIXES = ("sqlite_",)


@dataclass
class ColumnMeta:
    """Metadata for a single table column."""

    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool


@dataclass
class SchemaMetadata:
    """Metadata for a single database table."""

    table_name: str
    columns: list[ColumnMeta] = field(default_factory=list)
    foreign_keys: list[tuple[str, str, str]] = field(default_factory=list)
    # Each FK is (column_name, referenced_table, referenced_column)


def crawl_schema(
    engine: Engine,
    schema: str | None = None,
    exclude_tables: frozenset[str] = frozenset(),
) -> list[SchemaMetadata]:
    """
    Crawl all user-facing tables in the business database.

    Args:
        engine:         SQLAlchemy engine connected to the business database.
        schema:         Optional schema name (PostgreSQL). None uses the default.
        exclude_tables: Extra table names to skip (from configuration).

    Returns:
        One SchemaMetadata per table with full column and FK information.
    """
    inspector = inspect(engine)
    table_names = [
        name
        for name in inspector.get_table_names(schema=schema)
        if name not in _EXCLUDED_TABLES
        and name not in exclude_tables
        and not name.startswith(_EXCLUDED_PREFIXES)
    ]
    logger.info("Crawling schema for %d tables: %s", len(table_names), table_names)

    schemas: list[SchemaMetadata] = []
    for table_name in sorted(table_names):
        pk_constraint = inspector.get_pk_constraint(table_name, schema=schema)
        pk_columns = frozenset(pk_constraint.get("constrained_columns") or [])
        columns = _get_columns(inspector, table_name, schema, pk_columns)
        foreign_keys = _get_foreign_keys(inspector, table_name, schema)
        schemas.append(
            SchemaMetadata(
                table_name=table_name,
                columns=columns,
                foreign_keys=foreign_keys,
            )
        )

    return schemas


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_columns(
    inspector, table_name: str, schema: str | None, pk_columns: frozenset[str]
) -> list[ColumnMeta]:
    """Return ordered column metadata for a table."""
    return [
        ColumnMeta(
            name=col["name"],
            data_type=str(col["type"]),
            is_nullable=bool(col.get("nullable", True)),
            is_primary_key=(col["name"] in pk_columns),
        )
        for col in inspector.get_columns(table_name, schema=schema)
    ]


def _get_foreign_keys(
    inspector, table_name: str, schema: str | None
) -> list[tuple[str, str, str]]:
    """Return (column, referenced_table, referenced_column) tuples for all FKs."""
    foreign_keys: list[tuple[str, str, str]] = []
    for fk in inspector.get_foreign_keys(table_name, schema=schema):
        for col_name, ref_col in zip(
            fk.get("constrained_columns") or [], fk.get("referred_columns") or []
        ):
            foreign_keys.append((col_name, fk["referred_table"], ref_col))
    return foreign_keys
