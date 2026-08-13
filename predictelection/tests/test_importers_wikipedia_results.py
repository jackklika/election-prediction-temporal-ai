"""The results importer, against the Wisconsin page it was written from.

`contest_result` had no writer before this — the predicate backtesting needs.
The Wisconsin Democratic gubernatorial primary is a good first page precisely
because it punishes the obvious shortcut: the winner had previously withdrawn
from the race, and a withdrawn candidate out-polled two who stayed in. So `won`
comes from the page's own "Nominee" heading, and vote order is never consulted.
"""

from __future__ import annotations

from decimal import Decimal
import gzip
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection.importers import run_import
from predictelection.importers.wikipedia_polls import _PageWalker
from predictelection.importers.wikipedia_results import (
    WikipediaResultsImporter,
    candidate_name,
    nominee_names,
    parse_share,
    parse_votes,
)
from predictelection.sql import Claim, Entity, get_predicate_spec
from predictelection.tests.helpers import assert_reingestion_is_idempotent


FIXTURES = Path(__file__).parent / "fixtures"
WISCONSIN = "ocd-division/country:us/state:wi"
WI_CONTEST = f"{WISCONSIN}/governor/2026/primary/democratic"


def _page(name: str) -> bytes:
    return gzip.decompress((FIXTURES / f"{name}.html.gz").read_bytes())


def _importer() -> WikipediaResultsImporter:
    return WikipediaResultsImporter(
        url="https://en.wikipedia.org/wiki/2026_Wisconsin_gubernatorial_election",
        division=WISCONSIN,
        office="governor",
        cycle=2026,
    )


# ---------------------------------------------------------------------------
# Parsing — no database


def test_the_wisconsin_results_parse_exactly() -> None:
    """Seven candidates, pinned to the page. Counts are the whole product."""

    parsed = _importer().parse(_page("wikipedia_wi_governor_2026"))
    rows = {row.data["name"]: row.data for row in parsed.rows}

    assert len(parsed.rows) == 7
    assert rows["David Crowley"]["votes"] == "315278"
    assert rows["David Crowley"]["share"] == "39.81"
    assert rows["Francesca Hong"]["votes"] == "311495"
    assert rows["Sara Rodriguez"]["votes"] == "32687"


def test_won_comes_from_the_nominee_heading_not_vote_order() -> None:
    """The trap this page sets, and the reason `won` is read from prose.

    Crowley won *and* had previously withdrawn; Rodriguez withdrew *and*
    out-polled two candidates who stayed in. Sorting by votes would still get
    Crowley right here by luck, but the nominee list is what actually states it.
    """

    parsed = _importer().parse(_page("wikipedia_wi_governor_2026"))
    won = {row.data["name"]: row.data["won"] for row in parsed.rows}

    assert won["David Crowley"] == "true"
    assert won["Francesca Hong"] == "false"
    assert won["Sara Rodriguez"] == "false"  # withdrawn, and beat two who didn't


def test_the_nominee_list_is_read_from_the_roster() -> None:
    walker = _PageWalker()
    walker.feed(_page("wikipedia_wi_governor_2026").decode("utf-8", "replace"))
    assert nominee_names(walker, "Democratic primary") == {"David Crowley"}


def test_a_withdrawn_suffix_is_ballot_metadata_not_a_name() -> None:
    """ "Sara Rodriguez (withdrawn)" is one person, not a second one.

    Stripped for resolution so her result joins her candidacy; preserved in the
    printed name so review sees the page's own wording.
    """

    parsed = _importer().parse(_page("wikipedia_wi_governor_2026"))
    rodriguez = next(r for r in parsed.rows if r.data["name"] == "Sara Rodriguez")
    assert rodriguez.data["printed_name"] == "Sara Rodriguez (withdrawn)"


def test_totals_and_write_ins_are_skipped_not_parsed() -> None:
    parsed = _importer().parse(_page("wikipedia_wi_governor_2026"))
    names = {row.data["name"] for row in parsed.rows}
    assert not any(n.lower().startswith(("total", "write")) for n in names)
    assert parsed.skipped >= 2


def test_it_generalizes_to_another_page() -> None:
    """Michigan's 2026 Senate primaries, read without any Wisconsin-specific code.

    Both parties' nominees come from their own sections' rosters, which is the
    check that section scoping works: one page, two primaries, two winners.
    """

    michigan = WikipediaResultsImporter(
        url="https://en.wikipedia.org/wiki/fixture",
        division="ocd-division/country:us/state:mi",
        office="us-senate",
        cycle=2026,
    )
    rows = michigan.parse(_page("wikipedia_mi_senate_2026")).rows
    winners = {
        (r.data["party"], r.data["name"]) for r in rows if r.data["won"] == "true"
    }
    assert winners == {
        ("democratic", "Abdul El-Sayed"),
        ("republican", "Mike Rogers"),
    }


def test_an_incumbent_annotation_does_not_cost_a_winner_his_win() -> None:
    """Georgia prints "Raphael Warnock (incumbent)" in the table.

    The roster prints him without it, so normalising only one side recorded the
    primary's winner as `won: false` — a false claim about a real person on a
    real page. Both sides go through `candidate_name`.
    """

    assert candidate_name("Raphael Warnock (incumbent)") == "Raphael Warnock"

    georgia = WikipediaResultsImporter(
        url="https://en.wikipedia.org/wiki/fixture",
        division="ocd-division/country:us/state:ga",
        office="us-senate",
        cycle=2022,
    )
    rows = georgia.parse(_page("wikipedia_ga_senate_2022")).rows
    warnock = next(r for r in rows if r.data["name"] == "Raphael Warnock")
    assert warnock.data["won"] == "true"


def test_a_section_with_no_roster_leaves_won_unstated() -> None:
    """A general election has a winner, not a *nominee*, so there is no roster.

    This is what nullable `won` bought: Georgia's general and runoff tables would
    otherwise have asserted `won: false` for every candidate in them, including
    Warnock, who won the runoff. "" here becomes SQL NULL — the source said
    nothing about the outcome.
    """

    georgia = WikipediaResultsImporter(
        url="https://en.wikipedia.org/wiki/fixture",
        division="ocd-division/country:us/state:ga",
        office="us-senate",
        cycle=2022,
    )
    rows = georgia.parse(_page("wikipedia_ga_senate_2022")).rows
    unstated = {r.data["stage"] for r in rows if r.data["won"] == ""}
    assert unstated == {"general", "runoff"}
    # and nothing in those sections was asserted as a loss
    assert not any(
        r.data["won"] == "false" for r in rows if r.data["stage"] in unstated
    )


@pytest.mark.parametrize(
    ("text", "expected"), [("315,278", 315278), ("271", 271), ("—", None), ("", None)]
)
def test_vote_forms(text: str, expected: int | None) -> None:
    assert parse_votes(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("39.81", Decimal("39.81")), ("100.0", Decimal("100.0")), ("—", None)],
)
def test_share_forms(text: str, expected: Decimal | None) -> None:
    assert parse_share(text) == expected


# ---------------------------------------------------------------------------
# End to end


pytestmark = pytest.mark.postgres


def _results(session: Session) -> dict[str, dict]:
    version = get_predicate_spec("contest_result").predicate_version_id
    return {
        name: value
        for name, value in session.execute(
            select(Entity.canonical_name, Claim.value)
            .join(Claim, Claim.subject_id == Entity.id)
            .where(Claim.predicate_version_id == version)
        )
    }


def test_results_land_as_contest_result_claims(session: Session, object_store) -> None:
    result = run_import(
        session, object_store, _importer(), raw=_page("wikipedia_wi_governor_2026")
    )
    assert result.rows_failed == 0
    assert result.rows_read == 7
    assert result.alignment == 1.0

    stored = _results(session)
    assert stored["David Crowley"] == {
        "votes": 315278,
        "share": "39.81",
        "place": None,
        "won": True,
    }
    assert stored["Francesca Hong"]["won"] is False
    assert stored["Sara Rodriguez"]["votes"] == 32687


def test_results_attach_to_the_contest_by_key(session: Session, object_store) -> None:
    """The join that makes a result correlatable with polls and candidacies."""

    from predictelection.sql import EntityIdentifier

    run_import(
        session, object_store, _importer(), raw=_page("wikipedia_wi_governor_2026")
    )
    version = get_predicate_spec("contest_result").predicate_version_id
    keys = set(
        session.scalars(
            select(EntityIdentifier.value)
            .join(Claim, Claim.object_id == EntityIdentifier.entity_id)
            .where(
                Claim.predicate_version_id == version,
                EntityIdentifier.namespace == "contest-key",
            )
        )
    )
    assert keys == {WI_CONTEST}


def test_reimporting_results_writes_nothing(session: Session, object_store) -> None:
    """Counts change between canvass and certification; an unchanged page does not."""

    assert_reingestion_is_idempotent(
        session,
        lambda: run_import(
            session, object_store, _importer(), raw=_page("wikipedia_wi_governor_2026")
        ),
    )


def test_a_corrected_count_is_a_second_claim(session: Session, object_store) -> None:
    """Election night → canvass is a supersession chain, never an edit.

    The original stays readable against the bytes that stated it, which is what
    lets a backtest ask what was known on the night.
    """

    page = _page("wikipedia_wi_governor_2026")
    run_import(session, object_store, _importer(), raw=page)
    session.flush()
    before = session.scalar(select(func.count(Claim.id)))

    corrected = page.replace(b"315,278", b"315,301")
    run_import(session, object_store, _importer(), raw=corrected)
    session.flush()

    assert session.scalar(select(func.count(Claim.id))) == (before or 0) + 1
    crowley = [
        value
        for name, value in session.execute(
            select(Entity.canonical_name, Claim.value)
            .join(Claim, Claim.subject_id == Entity.id)
            .where(
                Claim.predicate_version_id
                == get_predicate_spec("contest_result").predicate_version_id
            )
        )
        if name == "David Crowley"
    ]
    assert sorted(v["votes"] for v in crowley) == [315278, 315301]
