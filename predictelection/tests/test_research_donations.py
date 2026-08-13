"""Donations, and what adding a domain is supposed to cost.

Half of this file tests the domain. The other half tests the *seam*: that a
predicate seeded today is readable through `query` with no reader written for it,
and that a donation reaches the graph without any layer above `research/`
learning what a donation is. If those stop holding, the architecture claim in
`ingestion-roadmap.md` §2 has quietly become false.

The domain's own hazard is that money mentioning a candidate is not money for
them. An independent expenditure run against someone is a donation whose
`supporting` is false, and a model that treats every dollar as support gets the
sign of the whole funding picture wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection import query
from predictelection.research.donations import ScrapedDonation, ingest_donation
from predictelection.research.ingestion import IngestContext
from predictelection.research.scraped import ScrapedEntity
from predictelection.sql import (
    Claim,
    ContestStage,
    DonationKind,
    DonationValue,
    Entity,
    EntityKind,
    SourceKind,
    TimePrecision,
    get_predicate_spec,
)
from predictelection.tests.helpers import assert_reingestion_is_idempotent


pytestmark = pytest.mark.postgres

WI_GOV = "ocd-division/country:us/state:wi"


@pytest.fixture
def snapshot(session: Session, object_store):
    from predictelection.research.archive import SourceArchive

    return SourceArchive(session, object_store).observe(
        kind=SourceKind.WEB_PAGE,
        canonical_url="https://example.test/wi-finance",
        content=b"<html>filings</html>",
        media_type="text/html",
        retrieved_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


def _donation(**overrides: Any) -> ScrapedDonation:
    base: dict[str, Any] = {
        "source_url": "https://example.test/wi-finance",
        "donor": ScrapedEntity(name="Jane Backer"),
        "recipient": ScrapedEntity(name="David Crowley"),
        "amount": Decimal("2500"),
        "division_id": WI_GOV,
        "office": "governor",
        "cycle": 2026,
        "stage": ContestStage.PRIMARY,
        "party": "Democratic",
        "given_on": datetime(2026, 3, 14, tzinfo=UTC),
    }
    return ScrapedDonation(**(base | overrides))


def _context(session: Session, snapshot) -> IngestContext:
    return IngestContext(session=session, snapshot=snapshot)


def _claims(session: Session) -> int:
    version = get_predicate_spec("donated_to").predicate_version_id
    return (
        session.scalar(
            select(func.count(Claim.id)).where(Claim.predicate_version_id == version)
        )
        or 0
    )


# ------------------------------------------------------------------ domain


def test_a_donation_lands_as_a_claim_about_the_donor(
    session: Session, snapshot
) -> None:
    """Subject is the giver. That is what keeps one shape for a person maxing
    out, a PAC bundling, and a super PAC spending against someone."""

    ingest_donation(_donation(), _context(session, snapshot))
    session.flush()

    (row,) = query.claims_with(session, "donated_to")
    assert row.subject.name == "Jane Backer"
    assert row.object is not None and row.object.name == "David Crowley"
    # The predicate is the discriminator, and narrowing on it is what the union
    # forces at every read site — including this one.
    assert isinstance(row.value, DonationValue)
    assert row.value.amount == Decimal("2500")
    assert row.value.kind is DonationKind.CONTRIBUTION
    # stored canonically, which is what makes an identical gift deduplicate
    assert row.raw_value is not None and row.raw_value["amount"] == "2500"
    assert row.valid_at is not None and row.valid_at.date().isoformat() == "2026-03-14"


def test_reingesting_the_same_donation_writes_nothing(
    session: Session, snapshot
) -> None:
    """The rule every predicate with a writer has to pass."""

    context = _context(session, snapshot)
    assert_reingestion_is_idempotent(
        session, lambda: ingest_donation(_donation(), context)
    )


def test_money_spent_against_someone_is_not_recorded_as_support(
    session: Session, snapshot
) -> None:
    """The failure that would invert the funding picture.

    An independent expenditure attacking a candidate mentions them exactly as
    often as one supporting them. Nothing about the shape of the record
    distinguishes the two, so the flag has to be carried and it has to be able
    to say "against".
    """

    context = _context(session, snapshot)
    ingest_donation(
        _donation(
            donor=ScrapedEntity(name="A Super PAC"),
            donor_kind=EntityKind.ORGANIZATION,
            amount=Decimal("1200000"),
            kind=DonationKind.INDEPENDENT_EXPENDITURE,
            supporting=False,
            purpose="television advertising",
        ),
        context,
    )
    session.flush()

    (row,) = query.claims_with(session, "donated_to")
    assert isinstance(row.value, DonationValue)
    assert row.value.kind is DonationKind.INDEPENDENT_EXPENDITURE
    assert row.value.supporting is False


def test_a_direct_contribution_leaves_support_unstated(
    session: Session, snapshot
) -> None:
    """Null rather than True: "supporting" is meaningless for a direct gift, and
    defaulting it either way would put an assertion in the graph nobody made.

    Present-and-null rather than absent, because `DonationValue` fills its own
    defaults — so there is exactly one way to say "unstated" and no caller has to
    tell a missing key from a null one.
    """

    ingest_donation(_donation(), _context(session, snapshot))
    session.flush()

    (row,) = query.claims_with(session, "donated_to")
    assert isinstance(row.value, DonationValue) and row.value.supporting is None


def test_an_undated_donation_is_stored_without_inventing_a_date(
    session: Session, snapshot
) -> None:
    """ "Maxed out to the campaign" is a real sentence with no date and no
    amount. Both absences are facts about the source."""

    ingest_donation(_donation(amount=None, given_on=None), _context(session, snapshot))
    session.flush()

    (row,) = query.claims_with(session, "donated_to")
    assert isinstance(row.value, DonationValue) and row.value.amount is None
    assert row.valid_at is None and row.valid_from is None


def test_a_vague_date_stays_vague(session: Session, snapshot) -> None:
    """The trap every domain in this repo shares: a month promoted to a day is
    an invented fact, and precision is the field that prevents it."""

    ingest_donation(
        _donation(
            given_on=datetime(2026, 7, 1, tzinfo=UTC),
            given_precision=TimePrecision.MONTH,
        ),
        _context(session, snapshot),
    )
    session.flush()

    (row,) = query.claims_with(session, "donated_to")
    assert row.valid_at_precision is TimePrecision.MONTH


def test_money_aimed_at_a_race_needs_no_candidate(session: Session, snapshot) -> None:
    """Ballot-measure spending has no person to attach to. The recipient is the
    contest itself, which is why this needed no second record type."""

    ingest_donation(
        _donation(
            recipient=ScrapedEntity(name="the governor's race"),
            recipient_kind=EntityKind.CONTEST,
        ),
        _context(session, snapshot),
    )
    session.flush()

    (row,) = query.claims_with(session, "donated_to")
    assert row.object is not None and row.object.kind is EntityKind.CONTEST


def test_two_donors_to_one_recipient_meet_on_one_entity(
    session: Session, snapshot
) -> None:
    """Funding and candidacy have to converge on the same person, or "who funded
    the winner" is unanswerable."""

    context = _context(session, snapshot)
    ingest_donation(_donation(), context)
    ingest_donation(
        _donation(
            donor=ScrapedEntity(name="Another PAC"),
            donor_kind=EntityKind.ORGANIZATION,
            amount=Decimal("5000"),
        ),
        context,
    )
    session.flush()

    assert _claims(session) == 2
    recipients = {
        row.object.entity_id
        for row in query.claims_with(session, "donated_to")
        if row.object
    }
    assert len(recipients) == 1
    assert (
        session.scalar(
            select(func.count(Entity.id)).where(
                Entity.canonical_name == "David Crowley"
            )
        )
        == 1
    )


# -------------------------------------------------------------------- seam


def test_the_read_surface_needed_no_donations_code(session: Session, snapshot) -> None:
    """The structural claim, stated as a test.

    `query` was written before this domain existed and contains the word
    "donation" nowhere. If a future domain cannot be read this way, that is the
    generic reader's gap — not a licence to add a per-domain query.
    """

    context = _context(session, snapshot)
    ingest_donation(_donation(), context)
    session.flush()

    donor = query.claims_with(session, "donated_to")[0].subject
    assert query.claims_about(session, donor.entity_id, predicate="donated_to")

    recipient_id = query.claims_with(session, "donated_to")[0].object
    assert recipient_id is not None
    received = query.claims_about(session, recipient_id.entity_id, as_object=True)
    assert [row.predicate for row in received] == ["donated_to"]


def test_a_donation_can_be_traced_to_its_source(session: Session, snapshot) -> None:
    """Citable or it does not count. The excerpt names both parties and the sum,
    so a reviewer sees what the claim asserts without opening the page."""

    ingest_donation(_donation(), _context(session, snapshot))
    session.flush()

    (row,) = query.claims_with(session, "donated_to")
    (cited,) = query.evidence_for(session, [row.claim_id])[row.claim_id]
    assert cited.source_url == "https://example.test/wi-finance"
    assert cited.excerpt is not None
    assert "Jane Backer" in cited.excerpt and "$2,500" in cited.excerpt


def test_the_ingest_activity_dispatches_a_donation_without_knowing_what_one_is(
    session: Session, snapshot
) -> None:
    """The registry seam. `ingestor_for` is how the activity stays domain-free,
    and a record in the union but not in INGESTORS raises rather than silently
    recording nothing."""

    from predictelection.research.registry import ingestor_for, payload_types

    record = _donation()
    assert type(record) in payload_types()
    assert ingestor_for(record) is ingest_donation
