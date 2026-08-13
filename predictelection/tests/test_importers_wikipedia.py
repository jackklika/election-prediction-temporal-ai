"""The Wikipedia poll importer, against the page it was written from.

The fixture is the real 2026 Michigan Senate article (gzipped, fetched
2026-08-12), not a hand-written table — hand-written fixtures are how the
namesake merge and the aggregation-table trap stayed invisible. When Wikipedia
changes its table conventions, re-fetch and watch what breaks; that is the
point of keeping the page real.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
import gzip
import json
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection.importers import run_import
from predictelection.importers.wikipedia_polls import (
    WikipediaPollsImporter,
    parse_fieldwork,
    parse_margin,
    parse_percentage,
    parse_sample,
)
from predictelection.sql import Poll, PollRevision
from predictelection.sql.polling import PollEstimate


PAGE = gzip.decompress(
    (
        Path(__file__).parent / "fixtures" / "wikipedia_mi_senate_2026.html.gz"
    ).read_bytes()
)

MICHIGAN = "ocd-division/country:us/state:mi"


def _importer() -> WikipediaPollsImporter:
    return WikipediaPollsImporter(
        url="https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Michigan",
        division=MICHIGAN,
        office="us-senate",
        cycle=2026,
    )


# ---------------------------------------------------------------------------
# Parsing the page — no database


def test_the_whole_page_parses_with_no_refusals() -> None:
    """Every kept row's every field must strict-parse.

    This is the claim the importer's design rests on: the deterministic tier
    covers real Wikipedia today, and the agent tier is for drift, not for the
    baseline. If this test starts failing after a re-fetch, that is Wikipedia
    moving — extend the parsers or route the new form to review; never loosen
    a parser into guessing.
    """

    parsed = _importer().parse(PAGE)
    assert len(parsed.rows) == 62
    for row in parsed.rows:
        assert parse_fieldwork(row.data["dates"]) is not None, row.data["dates"]
        for block in json.loads(row.data["samples"]):
            assert parse_sample(block["sample"]) is not None, block["sample"]
            assert parse_margin(block["moe"]) != (), block["moe"]
            for key, value in block.items():
                if key.startswith("reading:"):
                    parse_percentage(value)  # raises on junk


def test_sections_map_to_stage_and_party() -> None:
    rows = _importer().parse(PAGE).rows
    sections = {(row.data["stage"], row.data["party"]) for row in rows}
    assert sections == {
        ("primary", "democratic"),
        ("primary", "republican"),
        ("general", ""),
    }


def test_a_known_row_is_read_exactly() -> None:
    """One poll checked against the page, value for value."""

    first = _importer().parse(PAGE).rows[0]
    assert first.data["pollster"] == "SoCal Strategies (R)"
    assert parse_fieldwork(first.data["dates"]) == (
        date(2026, 7, 28),
        date(2026, 7, 30),
    )
    block = json.loads(first.data["samples"])[0]
    assert parse_sample(block["sample"]) == (437, "lv")
    assert parse_margin(block["moe"]) == Decimal("4.7")
    assert parse_percentage(block["reading:Abdul El-Sayed"]) == Decimal("56")
    # An em-dash means not offered in this question, not zero.
    assert parse_percentage(block["reading:Mallory McMorrow"]) is None


def test_aggregation_tables_are_excluded() -> None:
    """270toWin and Decision Desk HQ rows are averages, not polls.

    Importing them would double-count every poll they summarize. They fail
    header recognition ("Source of poll aggregation" is not "Poll source"),
    and their absence here is the assertion.
    """

    rows = _importer().parse(PAGE).rows
    pollsters = {row.data["pollster"] for row in rows}
    assert not any("270toWin" in name for name in pollsters)
    assert not any("Decision Desk" in name for name in pollsters)


def test_event_marker_rows_are_skipped_not_failed() -> None:
    parsed = _importer().parse(PAGE)
    for row in parsed.rows:
        assert "suspends" not in row.data["pollster"]
    assert parsed.skipped >= 4  # 2 aggregation tables + marker rows at least


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("July 28–30, 2026", (date(2026, 7, 28), date(2026, 7, 30))),
        ("April 11–13, 2026", (date(2026, 4, 11), date(2026, 4, 13))),
        ("June 29 – July 29, 2026", (date(2026, 6, 29), date(2026, 7, 29))),
        ("May 5, 2026", (date(2026, 5, 5), date(2026, 5, 5))),
        ("December 28 – January 3, 2026", (date(2025, 12, 28), date(2026, 1, 3))),
        ("through July 30, 2026", None),
        ("July 2026", None),
        ("", None),
    ],
)
def test_fieldwork_forms(text: str, expected) -> None:
    assert parse_fieldwork(text) == expected


# ---------------------------------------------------------------------------
# End to end — database


pytestmark = pytest.mark.postgres


def test_the_page_imports_and_reimports_idempotently(
    session: Session, object_store
) -> None:
    result = run_import(session, object_store, _importer(), raw=PAGE)
    assert result.rows_failed == 0
    assert result.rows_read == 62

    polls = session.scalar(select(func.count(Poll.id))) or 0
    revisions = session.scalar(select(func.count(PollRevision.id))) or 0
    estimates = session.scalar(select(func.count(PollEstimate.id))) or 0
    assert polls == 62  # one Poll per key
    assert revisions == 62  # one reading of each; a second source adds more
    assert estimates > 300  # several readings per sample row

    again = run_import(session, object_store, _importer(), raw=PAGE)
    assert again.rows_failed == 0
    assert session.scalar(select(func.count(PollRevision.id))) == revisions
    assert session.scalar(select(func.count(PollEstimate.id))) == estimates


def test_polls_attach_to_the_contest_by_key(session: Session, object_store) -> None:
    run_import(session, object_store, _importer(), raw=PAGE)

    keyed = [
        key
        for key in session.scalars(
            select(Poll.external_id).where(Poll.external_namespace == "poll-key")
        )
        if key is not None
    ]
    assert keyed
    prefix = f"{MICHIGAN}/us-senate/2026/"
    assert all(key.startswith(prefix) for key in keyed)
    assert any("/primary/democratic/" in key for key in keyed)
    assert any("/general/" in key for key in keyed)


def test_one_polls_rows_do_not_disagree_with_themselves(
    session: Session, object_store
) -> None:
    """LV/RV rows and matchup tables are one poll's samples and questions.

    Stored ungrouped, they collide on one PollKey as fake "disagreeing
    sources" — the first live run filed 61 spurious reviews this way. Only
    genuinely unkeyable or lookalike concerns may remain after a clean import.
    """

    from predictelection.sql import ReviewTask

    run_import(session, object_store, _importer(), raw=PAGE)
    reasons = [r for r in session.scalars(select(ReviewTask.reason)) if r]
    assert not any("disagree" in reason for reason in reasons)
