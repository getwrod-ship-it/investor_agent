"""
app/rag/embedder.py

Qdrant embedder.
embed_and_store() embeds business schema and knowledge base chunks with the
LOCAL BGE-M3 model (HuggingFaceEmbeddings — see bge-embed-rerank-deploy.md)
and upserts them into two Qdrant collections:
  - text2sql_schema  (one vector per business table)
  - text2sql_docs    (chunks from the user's knowledge base documents)

The embedding model is a multi-GB local model: it is loaded once per process
(lru_cache on _get_embeddings) and stays resident on the GPU.

Idempotent: if the schema collection already contains vectors and the docs
collection exists, the function returns immediately. An EMPTY knowledge base
is a valid state — the docs collection is simply created empty.

Triggered from the FastAPI lifespan handler in app/main.py on every startup,
and on demand via POST /schema/refresh or `python -m app.embed_cli`.
"""

import logging
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.business_db import get_business_engine
from app.config import get_settings
from app.rag.chunker import chunk_schema
from app.rag.crawler import crawl_schema
from app.rag.doc_loader import load_knowledge_base

logger = logging.getLogger(__name__)

# Qdrant collection names
SCHEMA_COLLECTION = "text2sql_schema"
DOCS_COLLECTION = "text2sql_docs"

# BGE-M3 dense vector dimension
VECTOR_SIZE = 1024

# The schema collection is considered fresh once it holds at least one vector.
# The docs collection only needs to EXIST — an empty knowledge base is valid.
_MIN_SCHEMA_VECTORS = 1

# Embedding happens in batches to bound GPU memory use
_EMBED_BATCH_SIZE = 50


@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client from current settings."""
    settings = get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the local BGE-M3 embedding model once and reuse it process-wide.

    Constructing the model per call would reload multi-GB weights on every
    query — the cache is what keeps the model resident on the GPU.
    """
    settings = get_settings()
    logger.info(
        "Loading embedding model %r on %s ...",
        settings.embedding_model,
        settings.embedding_device,
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _collections_are_fresh(client: QdrantClient) -> bool:
    """
    Return True if schema collection has vectors and docs collection exists.

    Used as the idempotency guard so re-runs of embed_and_store() on an already
    populated Qdrant instance do not re-embed everything.

    NOTE: after switching embedding models (e.g. voyage → BGE-M3) the stored
    vectors are incompatible — delete the collections or POST /schema/refresh
    to force a re-embed.
    """
    try:
        existing = {c.name for c in client.get_collections().collections}
        if SCHEMA_COLLECTION not in existing or DOCS_COLLECTION not in existing:
            return False

        schema_count = (client.get_collection(SCHEMA_COLLECTION).points_count or 0)

        fresh = schema_count >= _MIN_SCHEMA_VECTORS
        if fresh:
            logger.info(
                "Qdrant collections already populated (schema=%d vector(s)) — skipping re-embed",
                schema_count,
            )
        return fresh
    except Exception as exc:
        logger.warning("Could not check Qdrant collection state: %s", exc)
        return False


def _ensure_collection(client: QdrantClient, name: str) -> None:
    """Create a Qdrant collection with COSINE distance if it does not already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'", name)


def _docs_to_points(
    docs: list[Document], embeddings_model: HuggingFaceEmbeddings, id_offset: int = 0
) -> list[PointStruct]:
    """Embed a list of LangChain Documents in batches and return Qdrant PointStructs."""
    points: list[PointStruct] = []
    for batch_start in range(0, len(docs), _EMBED_BATCH_SIZE):
        batch = docs[batch_start : batch_start + _EMBED_BATCH_SIZE]
        texts = [doc.page_content for doc in batch]
        vectors = embeddings_model.embed_documents(texts)

        for i, (doc, vector) in enumerate(zip(batch, vectors, strict=False)):
            payload: dict[str, Any] = {"text": doc.page_content, **doc.metadata}
            points.append(PointStruct(id=id_offset + batch_start + i, vector=vector, payload=payload))

    return points


def embed_and_store() -> None:
    """
    Crawl the business MySQL schema, load the knowledge base, and embed
    both into Qdrant with the local BGE-M3 model.

    - Connects to Qdrant using host/port from settings.
    - Creates collections if they do not exist.
    - Skips re-embedding if the collections are already fresh (idempotent).
    - Crawls the live business database schema for schema chunks.
    - Loads every markdown/text file from knowledge_base/ at the project root
      for doc chunks.
    """
    client = _get_qdrant_client()

    if _collections_are_fresh(client):
        return

    embeddings_model = _get_embeddings()
    settings = get_settings()

    # --- Schema collection (one vector per business table) ---
    schemas = crawl_schema(
        get_business_engine(),
        exclude_tables=settings.excluded_tables,
    )
    schema_docs = chunk_schema(schemas)
    _ensure_collection(client, SCHEMA_COLLECTION)
    schema_points = _docs_to_points(schema_docs, embeddings_model)
    if schema_points:
        client.upsert(collection_name=SCHEMA_COLLECTION, points=schema_points)
        logger.info("Upserted %d vectors into '%s'", len(schema_points), SCHEMA_COLLECTION)
    else:
        logger.warning("No schema chunks produced — '%s' left empty", SCHEMA_COLLECTION)

    # --- Docs collection (chunks from the user's knowledge base files) ---
    doc_chunks = load_knowledge_base()
    _ensure_collection(client, DOCS_COLLECTION)
    doc_points = _docs_to_points(doc_chunks, embeddings_model)
    if doc_points:
        client.upsert(collection_name=DOCS_COLLECTION, points=doc_points)
        logger.info("Upserted %d vectors into '%s'", len(doc_points), DOCS_COLLECTION)
    else:
        # An empty knowledge base is a valid state — Qdrant rejects upserting
        # an empty point list, so the collection is simply left empty.
        logger.info("Knowledge base is empty — '%s' created empty", DOCS_COLLECTION)
