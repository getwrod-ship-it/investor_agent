"""
app/embed_cli.py

Command-line entry point for the vector ingestion step:

    python -m app.embed_cli

Crawls the business database schema, loads everything from
backend/knowledge_base/, and embeds both into Qdrant. Safe to run repeatedly —
idempotent unless the collections are missing or empty.

Inside Docker use:
    docker compose exec backend python -m app.embed_cli
"""

import logging

from app.rag.embedder import embed_and_store


def main() -> None:
    """Run the crawl + embed pipeline with console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    embed_and_store()


if __name__ == "__main__":
    main()
