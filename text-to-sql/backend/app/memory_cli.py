"""
app/memory_cli.py

Long-term memory rebuild CLI (DESIGN_long_term_memory.md §4.8 / DR-L10).

mem0 stores are derived, lossy products of query_run — when a store is lost
or polluted it can be rebuilt from the lossless audit log:

    PYTHONPATH=backend python -m app.memory_cli rebuild            # episodic only
    PYTHONPATH=backend python -m app.memory_cli rebuild --facts    # + facts (LLM cost)
    PYTHONPATH=backend python -m app.memory_cli rebuild --since 2026-10-01
    PYTHONPATH=backend python -m app.memory_cli rebuild --dry-run

- Episodic rebuild is pure code (add infer=False, ms per row) and wipes the
  episodic store for the configured user first (delete_all).
- --facts additionally replays every question/answer pair through the facts
  pipeline (infer=True) — 1~2 LLM calls per turn, so it is opt-in.
- Unsafe turns are skipped (same write-time filtering as the live path).
"""

import argparse
import logging
from datetime import datetime

from sqlmodel import Session, select

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("app.memory_cli")


def _iter_query_runs(session: Session, since: str | None):
    from app.models.db import QueryRun

    statement = select(QueryRun).order_by(QueryRun.created_at.asc())  # type: ignore[attr-defined]
    if since:
        statement = statement.where(QueryRun.created_at >= since)  # type: ignore[attr-defined]
    yield from session.exec(statement)


def rebuild(facts: bool, since: str | None, dry_run: bool) -> None:
    from app.config import get_settings
    from app.db import get_engine
    from app.memory import format_episode, get_memory_service

    service = get_memory_service()
    if not hasattr(service, "_episodic"):
        logger.error("memory is disabled (MEMORY_ENABLED=false) — nothing to rebuild")
        return

    uid = get_settings().mem0_user_id

    with Session(get_engine()) as session:
        rows = list(_iter_query_runs(session, since))

    eligible = [row for row in rows if row.intent != "unsafe"]
    logger.info(
        "query_run rows: %d total, %d eligible (unsafe skipped)%s",
        len(rows),
        len(eligible),
        f", since={since}" if since else "",
    )
    if dry_run:
        for row in eligible[:10]:
            print(format_episode(row.question, row.intent, row.generated_sql,
                                 row.final_explanation, row.session_id,
                                 now=row.created_at or datetime.now()))
        print(f"... ({len(eligible)} rows would be replayed, dry-run)")
        return

    # --- episodic: wipe + verbatim replay (pure code, no LLM) ---
    service._episodic.delete_all(user_id=uid)
    for row in eligible:
        episode = format_episode(
            row.question, row.intent, row.generated_sql, row.final_explanation,
            row.session_id, now=row.created_at or datetime.now(),
        )
        service._episodic.add(
            [{"role": "user", "content": episode}],
            user_id=uid,
            infer=False,
            metadata={
                "session_id": row.session_id,
                "trace_id": row.trace_id,
                "intent": row.intent,
                "created_at": (row.created_at or datetime.now()).isoformat(timespec="seconds"),
            },
        )
    logger.info("episodic store rebuilt: %d episodes", len(eligible))

    # --- facts: opt-in replay through the extraction/consolidation pipeline ---
    if facts:
        logger.info("replaying %d turns through facts pipeline (LLM cost) ...", len(eligible))
        for i, row in enumerate(eligible, 1):
            messages = [{"role": "user", "content": row.question}]
            if row.final_explanation:
                messages.append({"role": "assistant", "content": row.final_explanation})
            try:
                service._facts.add(messages, user_id=uid, infer=True)
            except Exception as exc:  # noqa: BLE001 — keep rebuilding the rest
                logger.error("facts replay failed on row %s: %s", row.id, exc)
            if i % 20 == 0:
                logger.info("facts replay progress: %d/%d", i, len(eligible))
        logger.info("facts store rebuilt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-term memory rebuild from query_run")
    sub = parser.add_subparsers(dest="command", required=True)
    p_rebuild = sub.add_parser("rebuild", help="rebuild mem0 stores from query_run")
    p_rebuild.add_argument("--facts", action="store_true",
                           help="also replay the facts pipeline (LLM cost per turn)")
    p_rebuild.add_argument("--since", default=None,
                           help="only replay rows with created_at >= ISO date (e.g. 2026-10-01)")
    p_rebuild.add_argument("--dry-run", action="store_true", help="print replay plan only")
    args = parser.parse_args()

    if args.command == "rebuild":
        rebuild(facts=args.facts, since=args.since, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
