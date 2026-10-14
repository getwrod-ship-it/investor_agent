"""
app/rag/chunker.py

Schema chunker: converts each SchemaMetadata object into a descriptive text
chunk suitable for embedding into the text2sql_schema Qdrant collection.
One chunk per table; includes column names, types, nullability, and FK
relationships. Rich business-level table descriptions should be supplied via
the knowledge base documents instead of code.
"""

from langchain_core.documents import Document

from app.rag.crawler import SchemaMetadata

_DEFAULT_DESCRIPTION = "Business data table. See knowledge base documents for details."


def chunk_schema(schemas: list[SchemaMetadata]) -> list[Document]:
    """
    Convert a list of SchemaMetadata objects into LangChain Documents.

    Each Document represents one table with a rich text description of its
    columns, nullability, primary keys, and foreign key relationships.
    Metadata fields: source, table_name, chunk_index.
    """
    documents: list[Document] = []
    for schema in schemas:
        text = _build_chunk_text(schema)
        doc = Document(
            page_content=text,
            metadata={
                "source": "schema",
                "table_name": schema.table_name,
                "chunk_index": 0,
            },
        )
        documents.append(doc)
    return documents


def _build_chunk_text(schema: SchemaMetadata) -> str:
    """Render a SchemaMetadata instance as a human-readable text block."""
    description = _DEFAULT_DESCRIPTION
    lines: list[str] = [
        f"Table: {schema.table_name}",
        f"Description: {description}",
        "Columns:",
    ]

    for col in schema.columns:
        nullable_label = "NULLABLE" if col.is_nullable else "NOT NULL"
        pk_label = " — Primary key" if col.is_primary_key else ""
        fk_label = ""
        for fk_col, ref_table, ref_col in schema.foreign_keys:
            if fk_col == col.name:
                fk_label = f" — Foreign key → {ref_table}.{ref_col}"
                break
        lines.append(
            f"  - {col.name} ({col.data_type}, {nullable_label}){pk_label}{fk_label}"
        )

    if schema.foreign_keys:
        fk_parts = [
            f"{fk_col} → {ref_table}({ref_col})"
            for fk_col, ref_table, ref_col in schema.foreign_keys
        ]
        lines.append(f"Foreign keys: {', '.join(fk_parts)}")
    else:
        lines.append("Foreign keys: none")

    return "\n".join(lines)
