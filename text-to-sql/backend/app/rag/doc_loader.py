"""
app/rag/doc_loader.py

Knowledge base document loader.
load_knowledge_base() reads every markdown (.md) and plain text (.txt) file
from knowledge_base/ at the project root, splits them into chunks using
LangChain's RecursiveCharacterTextSplitter, and attaches source metadata
(filename, chunk_index) to each chunk.

This directory is the designated drop-in point for the user's own business
knowledge (KPI definitions, domain rules, SQL conventions, glossaries, ...).
An empty directory is a valid state — the loader simply returns [].
"""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# knowledge_base/ lives three levels up from this file, at the project root:
# text-to-sql/knowledge_base/
_KB_DIR = Path(__file__).resolve().parents[3] / "knowledge_base"

_KB_PATTERNS = ("*.md", "*.txt")

_CHUNK_SIZE = 400
_CHUNK_OVERLAP = 40


def load_knowledge_base(kb_dir: Path | None = None) -> list[Document]:
    """
    Load and chunk all markdown/text files from the knowledge base directory.

    Each chunk is a LangChain Document with metadata:
      - source: "docs"
      - filename: the file name (e.g. "kpi_definitions.md")
      - chunk_index: zero-based position of this chunk within its file

    Args:
        kb_dir: Override the default knowledge_base directory path (useful in tests).

    Returns:
        List of Document chunks from all knowledge base files ([] if empty).
    """
    directory = kb_dir or _KB_DIR
    kb_files = sorted(
        {path for pattern in _KB_PATTERNS for path in directory.glob(pattern)}
    )

    if not kb_files:
        logger.warning(
            "No knowledge base files found in %s — the docs collection will be empty. "
            "Add .md or .txt files there and re-run the embedding step.",
            directory,
        )
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    all_chunks: list[Document] = []
    for kb_file in kb_files:
        raw_text = kb_file.read_text(encoding="utf-8")
        chunks = splitter.split_text(raw_text)
        for idx, chunk_text in enumerate(chunks):
            all_chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": "docs",
                        "filename": kb_file.name,
                        "chunk_index": idx,
                    },
                )
            )
        logger.debug("Loaded %d chunks from %s", len(chunks), kb_file.name)

    logger.info(
        "Knowledge base loaded: %d files → %d total chunks", len(kb_files), len(all_chunks)
    )
    return all_chunks
