"""
app/memory — long-term memory package (mem0 OSS, facts + episodic).

See docs/DESIGN_long_term_memory.md. Importing this package does NOT import
mem0 or load the GPU embedding model — both happen lazily inside
get_memory_service() only when memory_enabled and first used.
"""

from app.memory.service import (
    FACT_EXTRACTION_PROMPT_ZH,
    MemoryService,
    NullMemoryService,
    RecallResult,
    format_episode,
    get_memory_service,
)

__all__ = [
    "FACT_EXTRACTION_PROMPT_ZH",
    "MemoryService",
    "NullMemoryService",
    "RecallResult",
    "format_episode",
    "get_memory_service",
]
