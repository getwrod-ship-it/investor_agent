"""
app/rag/retriever.py

Qdrant retriever.
retrieve_context(question, top_k=3) embeds the user question with the local
BGE-M3 model, searches both Qdrant collections concurrently for a candidate
pool, then reranks the merged candidates with the local BGE cross-encoder and
returns the top-k chunks.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from app.rag.embedder import DOCS_COLLECTION, SCHEMA_COLLECTION, _get_embeddings, _get_qdrant_client
from app.rag.reranker import rerank_chunks

logger = logging.getLogger(__name__)

# Candidates fetched per collection before reranking — the cross-encoder then
# keeps the best top_k chunks overall.
_CANDIDATE_MULTIPLIER = 3
_MIN_CANDIDATES = 8


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its relevance score."""

    text: str
    score: float          # reranker relevance after rerank, cosine similarity before
    source: str           # "schema" or "docs"
    metadata: dict        # full payload minus the text field


def retrieve_context(question: str, top_k: int = 3) -> list[RetrievedChunk]:
    """
    Embed the question, search both Qdrant collections for candidates, then
    rerank with the local cross-encoder.

    Fetches max(top_k * 3, 8) candidates per collection, merges them, reranks,
    and returns up to top_k chunks PER SOURCE (schema and docs each keep their
    quota — strong doc hits must never push the table structure out of the
    context, or SQL generation hallucinates column names).

    Args:
        question: The natural-language question to search against.
        top_k:    Number of chunks to keep per source after reranking.

    Returns:
        List of RetrievedChunk objects, best first.
    """
    candidate_k = max(top_k * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATES)

    embeddings_model = _get_embeddings()
    question_vector = embeddings_model.embed_query(question)
    client: QdrantClient = _get_qdrant_client()

    schema_hits: list[ScoredPoint] = []
    docs_hits: list[ScoredPoint] = []

    def search_schema() -> list[ScoredPoint]:
        return client.query_points(
            collection_name=SCHEMA_COLLECTION,
            query=question_vector,
            limit=candidate_k,
        ).points

    def search_docs() -> list[ScoredPoint]:
        return client.query_points(
            collection_name=DOCS_COLLECTION,
            query=question_vector,
            limit=candidate_k,
        ).points

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(search_schema): "schema",
            executor.submit(search_docs): "docs",
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                hits = future.result()
                if source == "schema":
                    schema_hits = hits
                else:
                    docs_hits = hits
            except Exception as exc:
                logger.error("Qdrant search failed for collection '%s': %s", source, exc)

    chunks: list[RetrievedChunk] = []
    for hit in schema_hits:
        payload = dict(hit.payload or {})
        text = payload.pop("text", "")
        chunks.append(RetrievedChunk(text=text, score=hit.score, source="schema", metadata=payload))

    for hit in docs_hits:
        payload = dict(hit.payload or {})
        text = payload.pop("text", "")
        chunks.append(RetrievedChunk(text=text, score=hit.score, source="docs", metadata=payload))

    chunks.sort(key=lambda c: c.score, reverse=True)

    # Rerank the full candidate pool, then keep up to top_k chunks PER SOURCE.
    reranked = rerank_chunks(question, chunks, top_k=len(chunks))
    by_source: dict[str, list[RetrievedChunk]] = {}
    for chunk in reranked:
        by_source.setdefault(chunk.source, []).append(chunk)

    selected: list[RetrievedChunk] = []
    for source_chunks in by_source.values():
        selected.extend(source_chunks[:top_k])
    selected.sort(key=lambda c: c.score, reverse=True)

    logger.debug(
        "retrieve_context: question=%r → %d schema + %d doc candidates → %d kept",
        question[:60],
        len(schema_hits),
        len(docs_hits),
        len(selected),
    )
    return selected
