"""The write path a scraper uses: content-addressed rows, and one claim facade.

Four tables are content-addressed — claim.fingerprint, evidence_anchor.fingerprint,
artifact.sha256, and alias identity — and all four raise on the second scrape of
the same thing. That is the correct behaviour and a sharp edge: every scraper hits
it on run two. get_or_create turns it into the intended no-op.

The important semantics are in record_claim_from_source: when the claim already
exists it still gets a *new assertion*. A second source saying the same thing is
corroboration, not a duplicate, which is the whole reason claims and assertions
are separate tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping, TypeVar
import uuid

from sqlalchemy import ColumnElement, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from predictelection.sql.base import (
    Base,
    RecordOrigin,
    TimePrecision,
    idempotency_key,
)
from predictelection.sql.claim import (
    Claim,
    ClaimAssertion,
    EvidenceAnchor,
    EvidenceLocator,
    EvidenceStance,
    new_claim,
    new_claim_assertion,
    new_evidence_anchor,
)
from predictelection.sql.predicate import PredicateSpec, PredicateValue


_Model = TypeVar("_Model", bound=Base)


def get_or_create(
    session: Session,
    instance: _Model,
    *,
    key: ColumnElement[bool],
) -> tuple[_Model, bool]:
    """Fetch the row matching key, inserting instance only if it is missing.

    Returns (row, created). The insert runs inside a savepoint so that losing a
    race leaves the outer transaction usable: two activities scraping the same
    fact concurrently will both miss the SELECT, and the loser recovers by
    re-reading rather than by poisoning the session.

    Deliberately ORM rather than INSERT ... ON CONFLICT: Core inserts bypass the
    before_insert hooks that derive fingerprints, and having two write paths that
    must agree on derivation is a worse problem than a savepoint.
    """

    model = type(instance)
    existing = session.scalars(select(model).where(key)).first()
    if existing is not None:
        return existing, False

    try:
        with session.begin_nested():
            session.add(instance)
            session.flush()
    except IntegrityError:
        if instance in session:
            session.expunge(instance)
        return session.scalars(select(model).where(key)).one(), False
    return instance, True


@dataclass(frozen=True, slots=True)
class Validity:
    """When a claim holds: a point, an interval, or neither.

    The database enforces point-XOR-interval and pairs each endpoint with its
    precision. Building those six columns by hand at every call site is how they
    get mismatched, so construct them here instead.
    """

    at: datetime | None = None
    at_precision: TimePrecision | None = None
    start: datetime | None = None
    start_precision: TimePrecision | None = None
    end: datetime | None = None
    end_precision: TimePrecision | None = None

    @classmethod
    def timeless(cls) -> Validity:
        return cls()

    @classmethod
    def on(cls, moment: datetime, precision: TimePrecision) -> Validity:
        return cls(at=moment, at_precision=precision)

    @classmethod
    def between(
        cls,
        start: datetime,
        end: datetime | None,
        precision: TimePrecision,
        *,
        end_precision: TimePrecision | None = None,
    ) -> Validity:
        return cls(
            start=start,
            start_precision=precision,
            end=end,
            end_precision=(end_precision or precision) if end is not None else None,
        )

    def as_claim_kwargs(self) -> dict[str, Any]:
        return {
            "valid_at": self.at,
            "valid_at_precision": self.at_precision,
            "valid_from": self.start,
            "valid_from_precision": self.start_precision,
            "valid_to": self.end,
            "valid_to_precision": self.end_precision,
        }


def get_or_create_claim(
    session: Session,
    *,
    predicate: PredicateSpec,
    subject_id: uuid.UUID,
    object_id: uuid.UUID | None = None,
    value: PredicateValue | Mapping[str, Any] | None = None,
    validity: Validity | None = None,
) -> tuple[Claim, bool]:
    """The same proposition from two scrapes must land on one row."""

    claim = new_claim(
        predicate=predicate,
        subject_id=subject_id,
        object_id=object_id,
        value=value,
        **(validity or Validity.timeless()).as_claim_kwargs(),
    )
    return get_or_create(session, claim, key=Claim.fingerprint == claim.fingerprint)


def get_or_create_evidence_anchor(
    session: Session,
    *,
    source_snapshot_id: uuid.UUID,
    locator: EvidenceLocator | Mapping[str, Any],
    excerpt: str | None = None,
) -> tuple[EvidenceAnchor, bool]:
    anchor = new_evidence_anchor(
        source_snapshot_id=source_snapshot_id,
        locator=locator,
        excerpt=excerpt,
    )
    return get_or_create(
        session, anchor, key=EvidenceAnchor.fingerprint == anchor.fingerprint
    )


def record_claim_from_source(
    session: Session,
    *,
    predicate: PredicateSpec,
    subject_id: uuid.UUID,
    object_id: uuid.UUID | None = None,
    value: PredicateValue | Mapping[str, Any] | None = None,
    validity: Validity | None = None,
    source_snapshot_id: uuid.UUID,
    locator: EvidenceLocator | Mapping[str, Any],
    excerpt: str | None = None,
    research_run_id: uuid.UUID | None = None,
    stance: EvidenceStance = EvidenceStance.SUPPORTS,
    origin: RecordOrigin = RecordOrigin.MODEL,
    asserted_by: str | None = None,
    confidence: Decimal | None = None,
) -> ClaimAssertion:
    """Record one extracted fact, with the evidence that supports it.

    This is the call a scraper makes, and the whole call is idempotent. Running
    the same extraction twice deduplicates the proposition and the evidence
    location and returns the original assertion; a *different* run asserting the
    same proposition gets its own assertion, because independent corroboration is
    real information.

    Idempotent rather than raising matters for Temporal specifically: a retried
    activity must be able to complete, and a unique violation would fail it
    forever.

    Ontology alignment is checked and queued for review by new_claim_assertion.
    """

    claim, _ = get_or_create_claim(
        session,
        predicate=predicate,
        subject_id=subject_id,
        object_id=object_id,
        value=value,
        validity=validity,
    )
    anchor, _ = get_or_create_evidence_anchor(
        session,
        source_snapshot_id=source_snapshot_id,
        locator=locator,
        excerpt=excerpt,
    )

    key = idempotency_key(
        "claim_assertion",
        research_run_id=research_run_id,
        claim_fingerprint=claim.fingerprint,
        evidence_anchor_fingerprint=anchor.fingerprint,
        stance=stance,
    )
    matches_key = ClaimAssertion.idempotency_key == key
    existing = session.scalars(select(ClaimAssertion).where(matches_key)).first()
    if existing is not None:
        return existing

    # The savepoint covers the ReviewTask new_claim_assertion may also add, so a
    # lost race rolls back both rather than orphaning a task.
    try:
        with session.begin_nested():
            assertion = new_claim_assertion(
                session,
                claim=claim,
                evidence_anchor=anchor,
                idempotency_key=key,
                stance=stance,
                origin=origin,
                research_run_id=research_run_id,
                asserted_by=asserted_by,
                confidence=confidence,
            )
            session.flush()
    except IntegrityError:
        return session.scalars(select(ClaimAssertion).where(matches_key)).one()
    return assertion
