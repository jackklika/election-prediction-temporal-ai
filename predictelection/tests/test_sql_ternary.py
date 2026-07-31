"""Ternary (QUALIFIED) claims: subject, object, and a payload together.

The alternative is reifying the relationship as its own entity, which roughly
doubles the graph and mints entities with no canonical name — unresolvable on a
re-scrape, which is the failure we already measured with debate titles. These
tests pin the properties that make ternary worth having instead.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from predictelection.sql import (
    Claim,
    ContestStage,
    EndorsementStrength,
    EntityKind,
    ParticipationRole,
    PredicateTarget,
    TimePrecision,
    Validity,
    get_or_create_claim,
    get_predicate_spec,
    new_claim,
    new_claim_supersession,
)
from predictelection.tests import factories as f


pytestmark = pytest.mark.postgres


def test_role_is_part_of_identity(session: Session) -> None:
    """The whole point: a moderator is not a debater.

    Both claims are (person, participated_in, event). Only the payload separates
    them, so it has to reach the fingerprint or they collapse into one row.
    """

    spec = get_predicate_spec("participated_in")
    person = f.make_entity(session, kind=EntityKind.PERSON)
    event = f.make_entity(session, kind=EntityKind.EVENT)

    debated, created_first = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=person.id,
        object_id=event.id,
        value={"role": ParticipationRole.CANDIDATE},
    )
    moderated, created_second = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=person.id,
        object_id=event.id,
        value={"role": ParticipationRole.MODERATOR},
    )

    assert created_first and created_second
    assert debated.id != moderated.id
    assert session.scalar(select(func.count(Claim.id))) == 2


def test_the_same_role_still_deduplicates(session: Session) -> None:
    """Ternary must not cost us idempotency."""

    spec = get_predicate_spec("participated_in")
    person = f.make_entity(session, kind=EntityKind.PERSON)
    event = f.make_entity(session, kind=EntityKind.EVENT)
    payload = {"role": ParticipationRole.CANDIDATE}

    first, created_first = get_or_create_claim(
        session, predicate=spec, subject_id=person.id, object_id=event.id, value=payload
    )
    second, created_second = get_or_create_claim(
        session, predicate=spec, subject_id=person.id, object_id=event.id, value=payload
    )

    assert created_first is True
    assert created_second is False
    assert first.id == second.id


def test_a_qualified_claim_needs_both_halves(session: Session) -> None:
    spec = get_predicate_spec("participated_in")
    person = f.make_entity(session, kind=EntityKind.PERSON)
    event = f.make_entity(session, kind=EntityKind.EVENT)

    with pytest.raises(ValueError, match="requires a value"):
        new_claim(predicate=spec, subject_id=person.id, object_id=event.id)
    with pytest.raises(ValueError, match="requires an object"):
        new_claim(
            predicate=spec,
            subject_id=person.id,
            value={"role": ParticipationRole.CANDIDATE},
        )


def test_the_database_enforces_the_qualified_shape(session: Session) -> None:
    """ck_claim_target_matches_payload, not just the Python guard."""

    spec = get_predicate_spec("participated_in")
    person = f.make_entity(session, kind=EntityKind.PERSON)
    event = f.make_entity(session, kind=EntityKind.EVENT)
    claim = new_claim(
        predicate=spec,
        subject_id=person.id,
        object_id=event.id,
        value={"role": ParticipationRole.CANDIDATE},
    )
    session.add(claim)
    session.flush()

    with pytest.raises(IntegrityError) as error:
        session.execute(update(Claim).where(Claim.id == claim.id).values(value=None))
    assert "ck_claim_target_matches_payload" in str(error.value)


def test_an_unknown_role_is_rejected(session: Session) -> None:
    """The value model is the contract; a hallucinated role fails validation."""

    spec = get_predicate_spec("participated_in")
    person = f.make_entity(session, kind=EntityKind.PERSON)
    event = f.make_entity(session, kind=EntityKind.EVENT)

    with pytest.raises(Exception, match="role"):
        new_claim(
            predicate=spec,
            subject_id=person.id,
            object_id=event.id,
            value={"role": "chief heckler"},
        )


def test_outcomes_are_recordable_and_supersedable(session: Session) -> None:
    """Backtesting needs results, and results change between count and canvass."""

    spec = get_predicate_spec("contest_result")
    assert spec.target_kind is PredicateTarget.QUALIFIED
    candidate = f.make_entity(session, kind=EntityKind.PERSON)
    contest = f.make_entity(session, kind=EntityKind.CONTEST)

    election_night, _ = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=candidate.id,
        object_id=contest.id,
        value={"votes": 412_331, "share": Decimal("48.7"), "place": 1, "won": True},
    )
    certified, _ = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=candidate.id,
        object_id=contest.id,
        value={"votes": 412_408, "share": Decimal("48.7"), "place": 1, "won": True},
    )
    session.flush()

    # a corrected count is a different proposition, ready to supersede
    assert election_night.id != certified.id
    assert election_night.value is not None
    assert election_night.value["share"] == "48.7"


def test_vote_share_scale_does_not_fork_a_result(session: Session) -> None:
    """48.7 and 48.70 are the same result; CanonicalDecimal keeps them one row."""

    spec = get_predicate_spec("contest_result")
    candidate = f.make_entity(session, kind=EntityKind.PERSON)
    contest = f.make_entity(session, kind=EntityKind.CONTEST)

    first, _ = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=candidate.id,
        object_id=contest.id,
        value={"share": Decimal("48.7")},
    )
    second, created = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=candidate.id,
        object_id=contest.id,
        value={"share": Decimal("48.70")},
    )

    assert created is False
    assert first.id == second.id


def test_a_primary_and_a_general_are_separate_contests(session: Session) -> None:
    """They share an office but not candidates, polls, or outcomes."""

    office = f.make_entity(session, kind=EntityKind.OFFICE)
    primary = f.make_entity(session, kind=EntityKind.CONTEST)
    general = f.make_entity(session, kind=EntityKind.CONTEST)
    party = f.make_entity(session, kind=EntityKind.PARTY)

    for contest, stage in (
        (primary, ContestStage.PRIMARY),
        (general, ContestStage.GENERAL),
    ):
        get_or_create_claim(
            session,
            predicate=get_predicate_spec("contest_stage"),
            subject_id=contest.id,
            value={"stage": stage},
        )
        get_or_create_claim(
            session,
            predicate=get_predicate_spec("contest_for_office"),
            subject_id=contest.id,
            object_id=office.id,
        )
    get_or_create_claim(
        session,
        predicate=get_predicate_spec("contest_party"),
        subject_id=primary.id,
        object_id=party.id,
    )
    get_or_create_claim(
        session,
        predicate=get_predicate_spec("advances_to"),
        subject_id=primary.id,
        object_id=general.id,
    )
    session.flush()

    # the office is the join between the two stages
    both = session.scalars(
        select(Claim.subject_id).where(
            Claim.predicate_version_id
            == get_predicate_spec("contest_for_office").predicate_version_id,
            Claim.object_id == office.id,
        )
    ).all()
    assert set(both) == {primary.id, general.id}


def test_an_assessment_is_a_fact_about_the_assessor(session: Session) -> None:
    """ "Crowley was weakest" is unverifiable; "Murphy said so" is citable."""

    spec = get_predicate_spec("assessed")
    critic = f.make_entity(session, kind=EntityKind.PERSON)
    subject = f.make_entity(session, kind=EntityKind.PERSON)

    claim, _ = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=critic.id,
        object_id=subject.id,
        value={"rating": "weakest", "basis": "debate performance"},
        validity=Validity.on(datetime(2026, 7, 29, tzinfo=UTC), TimePrecision.DAY),
    )
    session.flush()

    assert claim.subject_id == critic.id
    assert claim.object_id == subject.id
    assert claim.value is not None
    assert claim.value["rating"] == "weakest"
    # no phantom "assessment" entity was minted to hold the rating
    assert session.scalar(select(func.count(f.Entity.id))) == 2


def test_endorsement_strength_distinguishes_a_withdrawal(session: Session) -> None:
    spec = get_predicate_spec("endorsed")
    endorser = f.make_entity(session, kind=EntityKind.ORGANIZATION)
    endorsee = f.make_entity(session, kind=EntityKind.PERSON)
    moment = datetime(2026, 6, 1, tzinfo=UTC)

    backed, _ = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=endorser.id,
        object_id=endorsee.id,
        value={"strength": EndorsementStrength.FULL},
        validity=Validity.on(moment, TimePrecision.DAY),
    )
    pulled, _ = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=endorser.id,
        object_id=endorsee.id,
        value={"strength": EndorsementStrength.WITHDRAWN},
        validity=Validity.on(datetime(2026, 8, 1, tzinfo=UTC), TimePrecision.DAY),
    )
    session.flush()

    assert backed.id != pulled.id


def test_a_recount_supersedes_rather_than_edits(session: Session) -> None:
    """Election night to certified is a chain of claims, not edits to one row.

    The graph has to answer "what did we believe on the night", so the original
    count stays readable. new_claim_supersession derives the idempotency key
    from the two claims, so a retried correction is a no-op instead of tripping
    uq_claim_supersession_predecessor.
    """

    candidate = f.make_entity(session, kind=EntityKind.PERSON)
    contest = f.make_entity(session, kind=EntityKind.CONTEST)
    spec = get_predicate_spec("contest_result")

    election_night, _ = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=candidate.id,
        object_id=contest.id,
        value={"votes": 481_000, "share": Decimal("48.7"), "won": False},
    )
    certified, _ = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=candidate.id,
        object_id=contest.id,
        value={"votes": 482_113, "share": Decimal("48.75"), "won": True},
    )
    session.flush()
    assert certified.id != election_night.id

    link = new_claim_supersession(
        predecessor=election_night,
        successor=certified,
        created_by="mi-sos-canvass",
        reason="certified canvass replaced the election-night count",
    )
    session.add(link)
    session.flush()

    # the superseded claim is still there to be read
    assert session.get(Claim, election_night.id) is not None
    assert link.predecessor_claim_id == election_night.id
    assert link.successor_claim_id == certified.id

    # the same correction, computed again, keys the same — a retry, not a clash
    again = new_claim_supersession(
        predecessor=election_night,
        successor=certified,
        created_by="mi-sos-canvass",
        reason="certified canvass replaced the election-night count",
    )
    assert again.idempotency_key == link.idempotency_key


def test_a_supersession_must_say_who_and_why(session: Session) -> None:
    """ck_claim_supersession_audit_nonempty, refused before the round trip."""

    candidate = f.make_entity(session, kind=EntityKind.PERSON)
    contest = f.make_entity(session, kind=EntityKind.CONTEST)
    spec = get_predicate_spec("contest_result")
    first, _ = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=candidate.id,
        object_id=contest.id,
        value={"votes": 1, "won": False},
    )
    second, _ = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=candidate.id,
        object_id=contest.id,
        value={"votes": 2, "won": False},
    )
    session.flush()

    with pytest.raises(ValueError, match="who made it and why"):
        new_claim_supersession(
            predecessor=first, successor=second, created_by="", reason="typo"
        )
