"""Run an importer from the command line.

    make import-ocd
    make import-fec CYCLE=2026

Same wiring the worker uses — the application database and the object store —
but no Temporal: an import is one deterministic pass over one file, and the
idempotency lives in the data layer (artifact SHA, identifier resolution,
assertion keys), so there is nothing for a workflow engine to add. Re-running
after a crash is the retry story.

One transaction per import, deliberately. A file either lands or it does not;
a partially imported file that half-exists would look exactly like a source
that was missing half its data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from predictelection.clients.sqlalchemy_engine import SqlAlchemyEngineClient
from predictelection.importers import (
    FecCandidateImporter,
    Importer,
    ImportResult,
    OcdImporter,
    run_import,
)
from predictelection.storage import S3ObjectStore, local_minio_config


def _build(args: argparse.Namespace) -> Importer:
    if args.importer == "ocd":
        return OcdImporter()
    if args.importer == "fec":
        return FecCandidateImporter(cycle=args.cycle)
    if args.importer == "wikipedia-results":
        from predictelection.importers.wikipedia_results import WikipediaResultsImporter

        _require(args, "wikipedia-results")
        return WikipediaResultsImporter(
            url=args.url,
            division=args.division,
            office=args.office,
            cycle=args.cycle,
        )
    if args.importer == "wikipedia-polls":
        from predictelection.importers.wikipedia_polls import WikipediaPollsImporter

        _require(args, "wikipedia-polls")
        normalizer = None
        if args.normalize:
            from predictelection.importers.normalize import AnthropicCellNormalizer

            normalizer = AnthropicCellNormalizer()
        return WikipediaPollsImporter(
            url=args.url,
            division=args.division,
            office=args.office,
            cycle=args.cycle,
            normalizer=normalizer,
        )
    raise ValueError(args.importer)


def _require(args: argparse.Namespace, importer: str) -> None:
    """Both Wikipedia importers need the race coordinates the page cannot state."""

    for required in ("url", "division", "office"):
        if not getattr(args, required):
            raise SystemExit(f"{importer} requires --{required}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "importer",
        choices=["ocd", "fec", "wikipedia-polls", "wikipedia-results"],
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=2026,
        help="Election cycle — the even year of the election.",
    )
    parser.add_argument("--url", default=None, help="Race article URL.")
    parser.add_argument(
        "--division",
        default=None,
        help="OCD division the race is held in, e.g. ocd-division/country:us/state:mi",
    )
    parser.add_argument(
        "--office", default=None, help="Office slug, e.g. us-senate, governor."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Let a model rewrite cells the strict parsers refuse (needs "
            "ANTHROPIC_API_KEY). The rewrite is re-parsed strictly and the "
            "revision is marked origin=model. Off, refusals fail their row."
        ),
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Read a local copy instead of fetching. Archived identically.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    importer = _build(args)
    store = S3ObjectStore(local_minio_config())
    store.ensure_bucket()
    session_factory = SqlAlchemyEngineClient().session_factory

    with session_factory() as session, session.begin():
        result = run_import(
            session,
            store,
            importer,
            raw=args.file.read_bytes() if args.file else None,
        )
        _report(result)

    if result.rows_failed:
        # Landed, but incompletely — the log has each failing row's index.
        sys.exit(1)


def _report(result: ImportResult) -> None:
    print(f"rows imported     {result.rows_read - result.rows_failed}")
    print(f"rows filtered     {result.rows_skipped}")
    if result.rows_failed:
        print(f"rows FAILED       {result.rows_failed}  (see log; run exits 1)")
    print(f"entities touched  {len(result.entities_touched)}")
    print(f"claims created    {result.claims_created}")
    if result.recorded:
        unchanged = sum(
            1 for item in result.recorded if item.outcome.value == "unchanged"
        )
        corroborated = len(result.recorded) - result.claims_created - unchanged
        print(f"claims corroborated {corroborated}")
        print(f"claims unchanged  {unchanged}")
    if result.alignment is not None:
        print(f"alignment         {result.alignment}")


if __name__ == "__main__":
    main()
