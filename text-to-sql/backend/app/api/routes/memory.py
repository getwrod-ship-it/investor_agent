"""
app/api/routes/memory.py

Dev/debug endpoint for long-term memory observability — inspect what the
mem0 facts/episodic stores return for a question, with scores. Used for
threshold tuning and the rename acceptance walkthrough; not consumed by the
frontend.
"""

import logging

from fastapi import APIRouter, Query

from app.memory import get_memory_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/search")
def search_memory(
    q: str = Query(min_length=2, max_length=200, description="检索问题"),
    top_k: int = Query(default=5, ge=1, le=20),
) -> dict:
    """
    Raw two-leg mem0 search (facts + episodic) with scores.

    Returns {"facts": [...], "episodes": [...]} where each item carries
    memory/score/created_at. Empty legs when memory is disabled or the
    backend is unreachable (fail-open, same as the request path).
    """
    service = get_memory_service()
    return service.search_with_scores(q, top_k=top_k)
