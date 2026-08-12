"""Poll identity and dedup: one poll, however many sources report it.

The properties that matter, in the order they protect the graph:

1. Re-ingesting the same reading writes nothing (payload hash).
2. A second source with identical numbers lands on the same Poll and writes
   nothing (PollKey + payload hash together).
3. A second source with *different* numbers becomes a second revision plus a
   ReviewTask — disagreement is surfaced, never averaged or overwritten.
4. Fuzzy anything (pollster near-miss, fieldwork off by a day) flags and
   proceeds; it never merges.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection.research.contests import ContestKey, PollKey
from predictelection.research.ingestion import IngestContext
from predictelection.research.polls import (
    POLL_KEY_NAMESPACE,
    PollReading,
    ScrapedPoll,
    ingest_poll,
    resolve_pollster,
)
from predictelection.research.scraped import ScrapedEntity
from predictelection.sql import (
    EntityAlias,
    Poll,
    PollRevision,
    ReviewTask,
)
from predictelection.sql.polling import PollEstimate, PollOption


pytestmark = pytest.mark.postgres

MI_SENATE_DEM = "ocd-division/country:us/state:mi/us-senate/2026/primary/democratic"


@pytest.fixture
def snapshot(session: Session, object_store):
    from datetime import UTC, datetime

    from predictelection.research.archive import SourceArchive
    from predictelection.sql import SourceKind

    return SourceArchive(session, object_store).observe(
        kind=SourceKind.WEB_PAGE,
        canonical_url="https://en.wikipedia.org/wiki/2026_MI_Senate",
        content=b"<html>poll tables</html>",
        media_type="text/html",
        retrieved_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


def _poll(**overrides: Any) -> ScrapedPoll:
    base: dict[str, Any] = {
        "source_url": "https://en.wikipedia.org/wiki/2026_MI_Senate",
        "pollster": "EPIC-MRA",
        "contest": ScrapedEntity(
            name="Michigan Senate Democratic Primary 2026",
            contest_key=MI_SENATE_DEM,
        ),
        "fieldwork_started_on": date(2026, 7, 24),
        "fieldwork_ended_on": date(2026, 7, 28),
        "sample_size": 600,
        "margin_of_error": Decimal("4.0"),
        "population": "lv",
        "readings": (
            PollReading(label="Stevens", percentage=Decimal("34")),
            PollReading(label="El-Sayed", percentage=Decimal("32")),
            PollReading(label="Undecided", percentage=Decimal("34")),
        ),
    }
    return ScrapedPoll(**(base | overrides))


def _counts(session: Session) -> tuple[int, int, int]:
    return (
        session.scalar(select(func.count(Poll.id))) or 0,
        session.scalar(select(func.count(PollRevision.id))) or 0,
        session.scalar(select(func.count(ReviewTask.id))) or 0,
    )


def test_a_poll_lands_with_options_and_estimates(session: Session, snapshot) -> None:
    context = IngestContext(session=session, snapshot=snapshot)
    ingest_poll(_poll(), context)
    session.flush()

    assert _counts(session) == (1, 1, 0)
    assert session.scalar(select(func.count(PollOption.id))) == 3
    assert session.scalar(select(func.count(PollEstimate.id))) == 3
    stored = session.scalars(select(Poll)).one()
    assert stored.external_namespace == POLL_KEY_NAMESPACE
    assert stored.external_id == f"{MI_SENATE_DEM}/epic-mra/2026-07-28"


def test_reingesting_the_same_reading_writes_nothing(
    session: Session, snapshot
) -> None:
    """Rule 1 of the roadmap, applied to the poll tables."""

    context = IngestContext(session=session, snapshot=snapshot)
    ingest_poll(_poll(), context)
    session.flush()
    before = _counts(session)

    ingest_poll(_poll(), context)
    session.flush()
    assert _counts(session) == before


def test_a_second_outlet_with_the_same_numbers_is_a_noop(
    session: Session, snapshot
) -> None:
    """Cross-source dedup: the whole reason identity is derived."""

    context = IngestContext(session=session, snapshot=snapshot)
    ingest_poll(_poll(), context)
    session.flush()
    before = _counts(session)

    ingest_poll(_poll(source_url="https://ballotpedia.org/MI_Senate_2026"), context)
    session.flush()
    assert _counts(session) == before


def test_disagreeing_sources_become_two_revisions_and_a_review(
    session: Session, snapshot
) -> None:
    """Never average, never overwrite: keep both and ask a human."""

    context = IngestContext(session=session, snapshot=snapshot)
    ingest_poll(_poll(), context)
    session.flush()

    disagreeing = _poll(
        source_url="https://example.com/other",
        readings=(
            PollReading(label="Stevens", percentage=Decimal("35")),
            PollReading(label="El-Sayed", percentage=Decimal("31")),
            PollReading(label="Undecided", percentage=Decimal("34")),
        ),
    )
    ingest_poll(disagreeing, context)
    session.flush()

    polls, revisions, reviews = _counts(session)
    assert (polls, revisions) == (1, 2)
    assert reviews >= 1
    reasons = [r for r in session.scalars(select(ReviewTask.reason)) if r]
    assert any("disagree" in reason for reason in reasons)


def test_fieldwork_off_by_a_day_flags_a_possible_duplicate(
    session: Session, snapshot
) -> None:
    """Two sources rounding field dates differently is a disguised duplicate."""

    context = IngestContext(session=session, snapshot=snapshot)
    ingest_poll(_poll(), context)
    session.flush()

    shifted = _poll(fieldwork_ended_on=date(2026, 7, 29))
    ingest_poll(shifted, context)
    session.flush()

    polls, _, reviews = _counts(session)
    assert polls == 2  # flagged, not merged
    reasons = [r for r in session.scalars(select(ReviewTask.reason)) if r]
    assert any("possible duplicate" in reason for reason in reasons)


def test_a_poll_without_fieldwork_dates_is_stored_but_flagged(
    session: Session, snapshot
) -> None:
    """No identity means no cross-source dedup; that must be visible."""

    context = IngestContext(session=session, snapshot=snapshot)
    ingest_poll(_poll(fieldwork_started_on=None, fieldwork_ended_on=None), context)
    session.flush()

    stored = session.scalars(select(Poll)).one()
    assert stored.external_id is None
    reasons = [r for r in session.scalars(select(ReviewTask.reason)) if r]
    assert any("could not be keyed" in reason for reason in reasons)


def test_punctuation_variants_of_a_pollster_collapse(
    session: Session, snapshot
) -> None:
    """ "EPIC-MRA" and "EPIC MRA" are one organization via the slug alias."""

    context = IngestContext(session=session, snapshot=snapshot)
    first = resolve_pollster(context, "EPIC-MRA")
    second = resolve_pollster(context, "EPIC MRA")
    third = resolve_pollster(context, "EPIC/MRA")
    session.flush()

    assert first.created is True
    assert second.entity_id == first.entity_id
    assert third.entity_id == first.entity_id


def test_a_lookalike_pollster_is_flagged_not_merged(session: Session, snapshot) -> None:
    """Emerson College vs Emerson College Polling: a human's call."""

    context = IngestContext(session=session, snapshot=snapshot)
    ingest_poll(_poll(pollster="Emerson College"), context)
    session.flush()

    # Through ingest, not a prior resolve_pollster call — resolving is what
    # creates the entity, so a direct call here would consume the very
    # "new pollster" condition the ReviewTask is filed on.
    near = _poll(
        pollster="Emerson College Polling",
        fieldwork_started_on=date(2026, 6, 1),
        fieldwork_ended_on=date(2026, 6, 5),
    )
    ingest_poll(near, context)
    session.flush()

    emersons = session.scalars(
        select(EntityAlias.entity_id)
        .where(
            EntityAlias.normalized_name.in_(
                ["emerson college", "emerson college polling"]
            )
        )
        .distinct()
    ).all()
    assert len(emersons) == 2  # forked, not merged
    reasons = [r for r in session.scalars(select(ReviewTask.reason)) if r]
    assert any("resembles existing" in reason for reason in reasons)


def test_poll_key_round_trips() -> None:
    key = PollKey.build(
        contest=ContestKey.parse(MI_SENATE_DEM),
        pollster="EPIC-MRA",
        fieldwork_end=date(2026, 7, 28),
    )
    assert PollKey.parse(str(key)) == key
    assert str(key) == f"{MI_SENATE_DEM}/epic-mra/2026-07-28"


def test_a_poll_needs_a_contest_key(session: Session, snapshot) -> None:
    """A poll attached to nothing correlates with nothing."""

    context = IngestContext(session=session, snapshot=snapshot)
    unkeyed = _poll(contest=ScrapedEntity(name="Some Race"))
    with pytest.raises(ValueError, match="contest_key"):
        ingest_poll(unkeyed, context)
