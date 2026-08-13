"""The poll importer against six real races chosen to disagree with each other.

The corpus is diversity, not volume — each page stresses a convention the
others do not: a runoff (GA 2022), a top-two primary with no party sections
(CA 2024), a partyless blanket primary and ranked-choice tables (AK 2022), a
different office in the same cycle (MI governor), and an older cycle's table
style (TX 2018).

Every count here is **pinned exactly**. The target is never "parse everything";
it is *on every page, either parse correctly or refuse visibly*. A pinned
`skipped=10` becoming 11 is a failure worth reading; an unpinned one is the
silence every bug this project has caught so far lived in. When a re-fetched
page moves a number, that is Wikipedia moving — extend a parser or let the
refusal stand, never loosen a parser into guessing.

The mutation tests at the bottom corrupt the page deliberately and assert the
importer refuses. They test the defenses, not the happy path — a safety check
that silently stopped checking is how 154 townships became one entity.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path

import pytest

from predictelection.importers.wikipedia_polls import (
    WikipediaPollsImporter,
    parse_fieldwork,
    parse_margin,
    parse_percentage,
    parse_sample,
)


FIXTURES = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class Race:
    division: str
    office: str
    cycle: int
    rows: int
    skipped: int
    refusing: int
    """Rows at least one of whose cells strict parsing correctly refuses —
    approximate counts, month-only dates, ranked-choice artifacts. These rows
    fail (or go to the tier-1 normalizer); they are never silently imported."""

    sections: frozenset[tuple[str, str]] = frozenset()


CORPUS: dict[str, Race] = {
    "wikipedia_mi_senate_2026": Race(
        "ocd-division/country:us/state:mi",
        "us-senate",
        2026,
        rows=62,
        skipped=8,
        refusing=0,
        sections=frozenset(
            {("primary", "democratic"), ("primary", "republican"), ("general", "")}
        ),
    ),
    "wikipedia_ga_senate_2022": Race(
        "ocd-division/country:us/state:ga",
        "us-senate",
        2022,
        rows=107,
        skipped=3,
        refusing=3,
        # The runoff is the point of this fixture: a fourth stage, correctly
        # kept apart from the general it followed.
        sections=frozenset(
            {
                ("primary", "democratic"),
                ("primary", "republican"),
                ("general", ""),
                ("runoff", ""),
            }
        ),
    ),
    "wikipedia_ca_senate_2024": Race(
        "ocd-division/country:us/state:ca",
        "us-senate",
        2024,
        rows=45,
        skipped=2,
        refusing=0,
        # Top-two: one partyless primary, exactly as Californians vote in it.
        sections=frozenset({("primary", ""), ("general", "")}),
    ),
    "wikipedia_mi_governor_2026": Race(
        "ocd-division/country:us/state:mi",
        "governor",
        2026,
        rows=58,
        skipped=10,
        refusing=2,
        sections=frozenset(
            {("primary", "democratic"), ("primary", "republican"), ("general", "")}
        ),
    ),
    "wikipedia_tx_senate_2018": Race(
        "ocd-division/country:us/state:tx",
        "us-senate",
        2018,
        rows=51,
        skipped=0,
        refusing=0,
        sections=frozenset(
            {("primary", "democratic"), ("primary", "republican"), ("general", "")}
        ),
    ),
    "wikipedia_ak_house_special_2022": Race(
        "ocd-division/country:us/state:ak/cd:1",
        "us-house",
        2022,
        rows=6,
        skipped=1,
        refusing=4,
        # Ranked choice: the degrade-gracefully case. Most rows refuse on RCV
        # round artifacts ("2*", "BA") — visibly, never as imported nonsense.
        sections=frozenset({("primary", ""), ("general", "")}),
    ),
}


def _page(name: str) -> bytes:
    return gzip.decompress((FIXTURES / f"{name}.html.gz").read_bytes())


def _importer(race: Race, **overrides) -> WikipediaPollsImporter:
    return WikipediaPollsImporter(
        url="https://en.wikipedia.org/wiki/fixture",
        division=race.division,
        office=race.office,
        cycle=race.cycle,
        **overrides,
    )


def _row_refuses(row) -> bool:
    if parse_fieldwork(row.data["dates"]) is None:
        return True
    for block in json.loads(row.data["samples"]):
        if parse_sample(block["sample"]) is None:
            return True
        if isinstance(parse_margin(block["moe"]), tuple):
            return True
        for key, value in block.items():
            if key.startswith("reading:"):
                try:
                    parse_percentage(value)
                except ValueError:
                    return True
    return False


@pytest.mark.parametrize("name", CORPUS)
def test_counts_are_pinned(name: str) -> None:
    """Kept, skipped and refusing are exact. Drift is a diff, never silence."""

    race = CORPUS[name]
    parsed = _importer(race).parse(_page(name))
    refusing = sum(1 for row in parsed.rows if _row_refuses(row))
    assert (len(parsed.rows), parsed.skipped, refusing) == (
        race.rows,
        race.skipped,
        race.refusing,
    )


@pytest.mark.parametrize("name", CORPUS)
def test_sections_map_exactly(name: str) -> None:
    race = CORPUS[name]
    rows = _importer(race).parse(_page(name)).rows
    assert {(r.data["stage"], r.data["party"]) for r in rows} == race.sections


@pytest.mark.parametrize("name", CORPUS)
def test_every_kept_percentage_is_a_percentage(name: str) -> None:
    """No cell that strict-parses may hold an impossible value.

    This is the invariant golden spot-checks cannot give on an unchecked page:
    whatever the layout did, nothing outside 0–100 came through as a reading.
    """

    race = CORPUS[name]
    for row in _importer(race).parse(_page(name)).rows:
        for block in json.loads(row.data["samples"]):
            for key, value in block.items():
                if not key.startswith("reading:"):
                    continue
                try:
                    percentage = parse_percentage(value)
                except ValueError:
                    continue  # a counted refusal, covered by the pinned counts
                assert percentage is None or 0 <= percentage <= 100


# ---------------------------------------------------------------------------
# Mutations: corrupt the page, assert refusal


def test_a_renamed_meta_column_rejects_the_whole_table() -> None:
    """If "Margin of error" becomes something else, the table must vanish from
    the kept rows — not import with columns shifted one left."""

    race = CORPUS["wikipedia_tx_senate_2018"]
    page = _page("wikipedia_tx_senate_2018")
    mutated = page.replace(b"Margin<br", b"Wiggle<br").replace(
        b"Margin of error", b"Wiggle room"
    )
    parsed = _importer(race).parse(mutated)
    assert len(parsed.rows) < race.rows
    assert parsed.skipped > race.skipped


def test_injected_junk_percentages_fail_rows_rather_than_import() -> None:
    """A cell that stops being a number must become a row failure, not a value."""

    race = CORPUS["wikipedia_tx_senate_2018"]
    mutated = _page("wikipedia_tx_senate_2018").replace(b">32%<", b">3o%<")
    parsed = _importer(race).parse(mutated)
    refusing = sum(1 for row in parsed.rows if _row_refuses(row))
    assert refusing > race.refusing


def test_the_page_without_polling_sections_yields_nothing() -> None:
    """Renaming the Polling headings must drop everything, counted or not —
    tables outside a polling section are results, fundraising, endorsements."""

    race = CORPUS["wikipedia_tx_senate_2018"]
    mutated = _page("wikipedia_tx_senate_2018").replace(b">Polling<", b">Surveys<")
    parsed = _importer(race).parse(mutated)
    assert len(parsed.rows) == 0
