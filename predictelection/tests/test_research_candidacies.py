"""The 2026 Wisconsin Democratic gubernatorial primary, as a timeline.

David Crowley announced, withdrew and endorsed Sara Rodriguez, re-entered ten
days later, and won. Rodriguez withdrew after his return, stayed on the ballot,
and still out-polled two candidates who never left.

That story is the reason validity intervals exist, and these tests are the first
thing in this project to actually depend on them. The assertion that matters is
not "the claims were stored" but **"asking who was running on a date returns a
different answer for different dates"** — which is only possible if a withdrawal
was recorded as an interval ending rather than a row changing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection.research.candidacies import (
    CandidacyOutcome,
    CandidacyStint,
    ScrapedCandidacy,
    ingest_candidacy,
)
from predictelection.research.endorsements import (
    ScrapedEndorsement,
    ingest_endorsement,
)
from predictelection.research.ingestion import IngestContext
from predictelection.research.scraped import ScrapedEntity
from predictelection.sql import (
    Claim,
    ContestStage,
    EndorsementStrength,
    Entity,
    EntityKind,
    SourceKind,
    TimePrecision,
    get_predicate_spec,
)
from predictelection.tests.helpers import assert_reingestion_is_idempotent


pytestmark = pytest.mark.postgres

WISCONSIN = "ocd-division/country:us/state:wi"
RACE: dict[str, Any] = {
    "division_id": WISCONSIN,
    "office": "Governor",
    "cycle": 2026,
    "stage": ContestStage.PRIMARY,
    "party": "Democratic",
}

# The real dates, to the precision the article gives them.
CROWLEY_IN = datetime(2026, 3, 4, tzinfo=UTC)
CROWLEY_OUT = datetime(2026, 7, 6, tzinfo=UTC)
CROWLEY_BACK = datetime(2026, 7, 25, tzinfo=UTC)
RODRIGUEZ_OUT = datetime(2026, 7, 15, tzinfo=UTC)


@pytest.fixture
def snapshot(session: Session, object_store):
    from predictelection.research.archive import SourceArchive

    return SourceArchive(session, object_store).observe(
        kind=SourceKind.WEB_PAGE,
        canonical_url="https://en.wikipedia.org/wiki/2026_Wisconsin_gubernatorial_election",
        content=b"<html>wisconsin</html>",
        media_type="text/html",
        retrieved_at=datetime(2026, 8, 13, tzinfo=UTC),
    )


@pytest.fixture
def context(session: Session, snapshot) -> IngestContext:
    return IngestContext(session=session, snapshot=snapshot)


def _crowley() -> ScrapedCandidacy:
    """In, out, back in — two stints, which is the whole point."""

    return ScrapedCandidacy(
        source_url="https://en.wikipedia.org/wiki/2026_Wisconsin_gubernatorial_election",
        candidate=ScrapedEntity(name="David Crowley"),
        stints=(
            CandidacyStint(entered_on=CROWLEY_IN, left_on=CROWLEY_OUT),
            CandidacyStint(entered_on=CROWLEY_BACK),
        ),
        outcome=CandidacyOutcome.NOMINATED,
        **RACE,
    )


def _rodriguez() -> ScrapedCandidacy:
    return ScrapedCandidacy(
        source_url="https://en.wikipedia.org/wiki/2026_Wisconsin_gubernatorial_election",
        candidate=ScrapedEntity(name="Sara Rodriguez"),
        stints=(
            CandidacyStint(
                entered_on=datetime(2026, 2, 1, tzinfo=UTC), left_on=RODRIGUEZ_OUT
            ),
        ),
        outcome=CandidacyOutcome.WITHDREW,
        remained_on_ballot=True,
        **RACE,
    )


def _running_on(session: Session, moment: datetime) -> set[str]:
    """Who the graph says was a candidate on a date, from intervals alone.

    This is the query the whole design is for: no status column, no "current"
    flag — just claims whose validity window contains the moment.
    """

    version = get_predicate_spec("candidate_in").predicate_version_id
    return set(
        session.scalars(
            select(Entity.canonical_name)
            .join(Claim, Claim.subject_id == Entity.id)
            .where(
                Claim.predicate_version_id == version,
                Claim.valid_from <= moment,
                (Claim.valid_to.is_(None)) | (Claim.valid_to > moment),
            )
        )
    )


# ---------------------------------------------------------------------------
# Candidacy intervals


def test_a_reentry_is_two_claims_not_an_overwrite(
    session: Session, context: IngestContext
) -> None:
    """The in→out→in arc. One row updated in place would lose the middle."""

    result = ingest_candidacy(_crowley(), context)
    session.flush()

    version = get_predicate_spec("candidate_in").predicate_version_id
    claims = list(
        session.scalars(
            select(Claim)
            .where(Claim.predicate_version_id == version)
            .order_by(Claim.valid_from)
        )
    )
    assert len(claims) == 2
    assert [c.valid_from for c in claims] == [CROWLEY_IN, CROWLEY_BACK]
    assert claims[0].valid_to == CROWLEY_OUT
    assert claims[1].valid_to is None  # ran to the election
    assert len(result.recorded) == 2


def test_who_was_running_changes_with_the_date(
    session: Session, context: IngestContext
) -> None:
    """The acceptance test for the whole feature.

    Three dates, three different answers, derived only from interval claims:
    before Crowley left both were running; while he was out only Rodriguez was;
    after he returned and she left only he was.
    """

    ingest_candidacy(_crowley(), context)
    ingest_candidacy(_rodriguez(), context)
    session.flush()

    before = _running_on(session, datetime(2026, 6, 15, tzinfo=UTC))
    during = _running_on(session, datetime(2026, 7, 10, tzinfo=UTC))
    after = _running_on(session, datetime(2026, 8, 11, tzinfo=UTC))

    assert before == {"David Crowley", "Sara Rodriguez"}
    assert during == {"Sara Rodriguez"}  # Crowley out, Rodriguez still in
    assert after == {"David Crowley"}  # Crowley back, Rodriguez withdrawn


def test_a_withdrawal_is_not_a_deletion(
    session: Session, context: IngestContext
) -> None:
    """Rodriguez left the race and is still a candidate in May, forever."""

    ingest_candidacy(_rodriguez(), context)
    session.flush()

    assert "Sara Rodriguez" in _running_on(session, datetime(2026, 5, 1, tzinfo=UTC))
    assert "Sara Rodriguez" not in _running_on(
        session, datetime(2026, 8, 1, tzinfo=UTC)
    )


def test_a_vague_date_keeps_its_precision(
    session: Session, context: IngestContext
) -> None:
    """ "dropped out in mid-July" is a month, and must stay a month.

    Storing DAY precision here would let a later reader believe the source named
    a day it never named.
    """

    vague = ScrapedCandidacy(
        source_url="https://example.test/wi",
        candidate=ScrapedEntity(name="Missy Hughes"),
        stints=(
            CandidacyStint(
                entered_on=datetime(2026, 1, 1, tzinfo=UTC),
                entered_precision=TimePrecision.MONTH,
                left_on=datetime(2026, 7, 1, tzinfo=UTC),
                left_precision=TimePrecision.MONTH,
            ),
        ),
        outcome=CandidacyOutcome.WITHDREW,
        remained_on_ballot=True,
        **RACE,
    )
    ingest_candidacy(vague, context)
    session.flush()

    claim = session.scalars(
        select(Claim).where(
            Claim.predicate_version_id
            == get_predicate_spec("candidate_in").predicate_version_id
        )
    ).one()
    assert claim.valid_from_precision is TimePrecision.MONTH
    assert claim.valid_to_precision is TimePrecision.MONTH


def test_overlapping_stints_are_refused() -> None:
    """Usually a re-entry reported with the original announcement date.

    Refused at the model boundary because the resulting claims would assert
    someone was in the race twice at once, which no query could make sense of.
    """

    with pytest.raises(ValueError, match="overlaps"):
        ScrapedCandidacy(
            source_url="https://example.test/wi",
            candidate=ScrapedEntity(name="David Crowley"),
            stints=(
                CandidacyStint(entered_on=CROWLEY_IN, left_on=CROWLEY_OUT),
                CandidacyStint(entered_on=CROWLEY_IN),  # the mistake
            ),
            **RACE,
        )


def test_the_candidacy_writes_no_result_claim(
    session: Session, context: IngestContext
) -> None:
    """Only the results importer writes contest_result.

    Two writers gave Crowley a claim saying he won beside one whose `won`
    defaulted to False. The nominee outcome is recorded from the same page by
    the importer instead.
    """

    ingest_candidacy(_crowley(), context)
    session.flush()

    version = get_predicate_spec("contest_result").predicate_version_id
    assert (
        session.scalar(
            select(func.count(Claim.id)).where(Claim.predicate_version_id == version)
        )
        == 0
    )


def test_reingesting_a_candidacy_writes_nothing(
    session: Session, context: IngestContext
) -> None:
    assert_reingestion_is_idempotent(
        session, lambda: ingest_candidacy(_crowley(), context)
    )


# ---------------------------------------------------------------------------
# Endorsements


def _endorsement(**overrides: Any) -> ScrapedEndorsement:
    base: dict[str, Any] = {
        "source_url": "https://example.test/wi",
        "endorser": ScrapedEntity(name="David Crowley"),
        "endorsee": ScrapedEntity(name="Sara Rodriguez"),
        "announced_on": CROWLEY_OUT,
        "ended_on": CROWLEY_BACK,
        **RACE,
    }
    return ScrapedEndorsement(**(base | overrides))


def test_an_endorsement_and_its_withdrawal_are_two_claims(
    session: Session, context: IngestContext
) -> None:
    """The roadmap's Phase 4 acceptance criterion, verbatim.

    Crowley backed Rodriguez only while he was out of the race. Both claims stay
    retrievable with distinct validity, so "who had this endorsement in July"
    still answers correctly after it ended.
    """

    ingest_endorsement(_endorsement(), context)
    ingest_endorsement(
        _endorsement(
            strength=EndorsementStrength.WITHDRAWN,
            announced_on=CROWLEY_BACK,
            ended_on=None,
        ),
        context,
    )
    session.flush()

    version = get_predicate_spec("endorsed").predicate_version_id
    claims = list(
        session.scalars(
            select(Claim)
            .where(Claim.predicate_version_id == version)
            .order_by(Claim.valid_from)
        )
    )
    assert len(claims) == 2
    assert claims[0].value is not None and claims[0].value["strength"] == "full"
    assert claims[1].value is not None and claims[1].value["strength"] == "withdrawn"
    assert claims[0].valid_to == CROWLEY_BACK
    assert claims[1].valid_from == CROWLEY_BACK


def test_switching_endorsements_keeps_both(
    session: Session, context: IngestContext
) -> None:
    """Hughes endorsed Rodriguez, then Crowley. Both true, at different times."""

    ingest_endorsement(
        _endorsement(
            endorser=ScrapedEntity(name="Missy Hughes"),
            endorsee=ScrapedEntity(name="Sara Rodriguez"),
            announced_on=datetime(2026, 7, 7, tzinfo=UTC),
            ended_on=datetime(2026, 7, 26, tzinfo=UTC),
        ),
        context,
    )
    ingest_endorsement(
        _endorsement(
            endorser=ScrapedEntity(name="Missy Hughes"),
            endorsee=ScrapedEntity(name="David Crowley"),
            announced_on=datetime(2026, 7, 26, tzinfo=UTC),
            ended_on=None,
        ),
        context,
    )
    session.flush()

    version = get_predicate_spec("endorsed").predicate_version_id
    rows = list(
        session.execute(
            select(Entity.canonical_name, Claim.valid_from)
            .join(Claim, Claim.object_id == Entity.id)
            .where(Claim.predicate_version_id == version)
            .order_by(Claim.valid_from)
        )
    )
    assert [name for name, _ in rows] == ["Sara Rodriguez", "David Crowley"]


def test_an_organization_can_endorse(session: Session, context: IngestContext) -> None:
    """Newspapers and unions endorse too, and are not people."""

    ingest_endorsement(
        _endorsement(
            endorser=ScrapedEntity(name="Milwaukee Journal Sentinel"),
            endorser_kind=EntityKind.ORGANIZATION,
        ),
        context,
    )
    session.flush()

    endorser = session.scalars(
        select(Entity).where(Entity.canonical_name == "Milwaukee Journal Sentinel")
    ).one()
    assert endorser.kind is EntityKind.ORGANIZATION


def test_endorsements_are_ontology_aligned(
    session: Session, context: IngestContext
) -> None:
    """PERSON→PERSON and ORGANIZATION→PERSON are both in the declared domain.

    `endorsed` is the first predicate here written with a non-PERSON subject, so
    this is a real check rather than a formality — a misaligned claim would be
    stored and queued for review instead of counted.
    """

    recorded = [
        ingest_endorsement(_endorsement(), context).recorded[0],
        ingest_endorsement(
            _endorsement(
                endorser=ScrapedEntity(name="Wisconsin AFL-CIO"),
                endorser_kind=EntityKind.ORGANIZATION,
            ),
            context,
        ).recorded[0],
    ]
    session.flush()

    assert all(item.assertion.ontology_aligned for item in recorded)


def test_reingesting_an_endorsement_writes_nothing(
    session: Session, context: IngestContext
) -> None:
    assert_reingestion_is_idempotent(
        session, lambda: ingest_endorsement(_endorsement(), context)
    )


def test_a_naive_date_from_an_agent_is_accepted_as_utc(
    session: Session, context: IngestContext
) -> None:
    """The failure that killed the first live run.

    The agent reported a withdrawal as "2026-07-06T00:00:00" with no offset and
    the whole workflow died on `valid_from must be timezone-aware`. A date has no
    offset to state, so demanding one invites an invented offset; the precision
    field already says the time component is not meaningful.
    """

    from datetime import datetime as naive_datetime

    record = ScrapedCandidacy(
        source_url="https://example.test/wi",
        candidate=ScrapedEntity(name="Joel Brennan"),
        stints=(
            CandidacyStint(
                entered_on=naive_datetime(2026, 3, 1),  # no tzinfo
                left_on=naive_datetime(2026, 8, 11),
            ),
        ),
        **RACE,
    )
    assert record.stints[0].entered_on is not None
    assert record.stints[0].entered_on.tzinfo is not None

    ingest_candidacy(record, context)
    session.flush()  # would raise "must be timezone-aware" before the fix

    assert "Joel Brennan" in _running_on(session, datetime(2026, 5, 1, tzinfo=UTC))
