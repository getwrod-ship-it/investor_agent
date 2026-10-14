"""
app/rag/reranker.py

Local BGE reranker (cross-encoder).
Wraps FlagReranker (BAAI/bge-reranker-v2-m3, loaded from the local directory
configured by RERANKER_MODEL — see bge-embed-rerank-deploy.md) and reorders
retrieved chunks by relevance to the question.

The model is loaded once per process and stays resident on the GPU
(milliseconds per rerank on GPU); it must NOT be constructed per request.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from FlagEmbedding import FlagReranker

from app.config import get_settings

if TYPE_CHECKING:  # avoid a circular import at runtime (retriever imports us)
    from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_reranker() -> FlagReranker:
    """Load the local reranker model once and reuse it process-wide."""
    settings = get_settings()
    logger.info(
        "Loading reranker model %r on %s ...",
        settings.reranker_model,
        settings.embedding_device,
    )
    return FlagReranker(
        settings.reranker_model,
        use_fp16=True,
        devices=settings.embedding_device,
    )


def rerank_chunks(
    question: str, chunks: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    """
    Rerank candidate chunks against the question with the local cross-encoder.

    Args:
        question: The natural-language question.
        chunks:   Candidate chunks fetched from the vector store (any order;
                  expected in descending cosine order as a fallback).
        top_k:    Number of chunks to keep after reranking. Callers that need
                  per-source quotas pass top_k=len(chunks) and slice themselves.

    Returns:
        Up to top_k chunks, best first. Each returned chunk's score is the
        sigmoid-normalised reranker relevance; the original cosine similarity
        is preserved in metadata["cosine_score"]. If reranking fails, the
        input (cosine) order is returned truncated to top_k.
    """
    if len(chunks) <= 1:
        return chunks

    # Runtime import (not just TYPE_CHECKING): RetrievedChunk is constructed
    # below. Imported lazily because retriever imports this module at load time.
    from app.rag.retriever import RetrievedChunk

    try:
        reranker = _get_reranker()
        pairs = [[question, chunk.text] for chunk in chunks]
        scores = reranker.compute_score(pairs, normalize=True)
    except Exception as exc:
        logger.error("Reranker failed — falling back to cosine order: %s", exc)
        return chunks[:top_k]

    if isinstance(scores, float):  # single-pair return type
        scores = [scores]

    ranked = sorted(zip(chunks, scores, strict=False), key=lambda pair: pair[1], reverse=True)

    result: list[RetrievedChunk] = []
    for chunk, rerank_score in ranked[:top_k]:
        result.append(
            RetrievedChunk(
                text=chunk.text,
                score=float(rerank_score),
                source=chunk.source,
                metadata={**chunk.metadata, "cosine_score": chunk.score},
            )
        )

    logger.debug("rerank_chunks: %d candidates → top %d", len(chunks), len(result))
    return result
