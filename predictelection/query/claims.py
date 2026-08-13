"""The generic claim reader — one query shape for every predicate.

This is where the ontology stops being a design and starts paying. Donations,
endorsements, candidacies and results are the same row shape with a different
predicate slug, so a reader that takes the slug as an argument covers a domain
that has not been written yet. A new domain needs a `PredicateSpec`, an ingestor,
and *nothing here*.

What a caller always needs and a raw claim row never has: the names behind the
subject and object ids, the predicate slug behind the version id, and whether
anyone has reviewed it. Joining those per row is the thing everybody writes by
hand and gets subtly wrong — `scripts/wi_timeline.py` did it four times over,
slightly differently each time, before it was ported onto this.

Provenance is summarised rather than expanded. A claim can carry many assertions
(that is the point of the split — a second source is corroboration), so a reader
gets the count and the distinct sources; a caller that needs the anchors asks for
one claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any
import uuid

from pydantic import ValidationError
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from predictelection.sql.base import TimePrecision
from predictelection.sql.claim import Claim, ClaimAssertion, EvidenceAnchor
from predictelection.sql.entity import Entity, EntityKind
from predictelection.sql.predicate import (
    ClaimValue,
    get_predicate_spec,
    get_predicate_spec_by_id,
    parse_claim_value,
)
from predictelection.sql.provenance import Source, SourceSnapshot


logger = logging.getLogger(__name__)


DEFAULT_LIMIT = 100
"""Higher than the agent-facing default: a page of results is not a prompt."""


@dataclass(frozen=True, slots=True)
class EntityRef:
    """An entity as a reader needs it: id to link, name to show."""

    entity_id: uuid.UUID
    kind: EntityKind
    name: str


@dataclass(frozen=True, slots=True)
class Evidence:
    """One assertion's citation: who said it, where, and what the page said.

    `excerpt` is nullable because not every anchor carries one — a table row
    locator points at a cell without quoting it — and inventing a quote to fill
    the gap would be worse than showing the URL alone.
    """

    asserted_by: str
    excerpt: str | None
    source_url: str
    source_title: str | None
    retrieved_at: datetime


@dataclass(frozen=True, slots=True)
class ClaimRow:
    """One claim, readable without another query.

    `valid_from`/`valid_to` are the interval form and `valid_at` the point form;
    a predicate uses one or the other, never both, and `ck_claim_point_or_interval`
    enforces it. Precision travels with each, because a `MONTH`-precision date
    rendered as a day is an invention the storage layer went out of its way to
    avoid.
    """

    claim_id: uuid.UUID
    predicate: str
    """The discriminator for `value`. None of the payload models carries a type
    tag and none can gain one — `build_claim_fingerprint` hashes the value, so a
    tag would change the identity of every stored claim — so this field is how a
    reader, or a TypeScript client, narrows the union."""

    subject: EntityRef
    object: EntityRef | None
    value: ClaimValue | None
    """Parsed into the model its predicate declares, not a bare dict.

    None both when the predicate takes no value and when a stored payload failed
    to validate against its own contract; `raw_value` still holds the payload in
    the second case, and the failure is logged rather than raised so one bad row
    cannot fail a page."""

    raw_value: dict[str, Any] | None = None
    """The payload as stored. The escape hatch for a claim written under an
    older version of its predicate's schema."""

    valid_at: datetime | None = None
    valid_at_precision: TimePrecision | None = None
    valid_from: datetime | None = None
    valid_from_precision: TimePrecision | None = None
    valid_to: datetime | None = None
    valid_to_precision: TimePrecision | None = None

    assertion_count: int = 0
    """How many times this proposition has been asserted — corroboration, not
    duplication. One is a single source; more means independent agreement."""

    asserted_by: tuple[str, ...] = ()
    """Distinct writers, so a claim only the agent believes is tellable from one
    an importer and the agent both wrote."""

    @property
    def is_open(self) -> bool:
        """An interval claim with no end: still true as far as anyone recorded."""

        return self.valid_from is not None and self.valid_to is None


def claims_about(
    session: Session,
    entity_id: uuid.UUID,
    *,
    predicate: str | None = None,
    as_object: bool = False,
    at: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> tuple[ClaimRow, ...]:
    """Everything the graph asserts about one entity.

    `as_object` flips the direction, and both directions matter: a person's
    endorsements are claims where they are the *subject*, while the endorsements
    they received are claims where they are the object. A reader that only ever
    looked at one side would show half a profile.
    """

    column = Claim.object_id if as_object else Claim.subject_id
    statement = select(Claim).where(column == entity_id)
    return _rows(session, _narrow(statement, predicate=predicate, at=at), limit=limit)


def claims_with(
    session: Session,
    predicate: str,
    *,
    subject_ids: Sequence[uuid.UUID] | None = None,
    object_ids: Sequence[uuid.UUID] | None = None,
    at: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> tuple[ClaimRow, ...]:
    """Every claim under one predicate, optionally narrowed to some entities.

    The shape a new domain gets for free: `claims_with(session, "donated_to",
    object_ids=[committee])` works the day the predicate is seeded, with no
    reader written for it.
    """

    statement = select(Claim)
    if subject_ids is not None:
        statement = statement.where(Claim.subject_id.in_(subject_ids))
    if object_ids is not None:
        statement = statement.where(Claim.object_id.in_(object_ids))
    return _rows(session, _narrow(statement, predicate=predicate, at=at), limit=limit)


# --------------------------------------------------------------------------


def _narrow(
    statement: Select[tuple[Claim]],
    *,
    predicate: str | None,
    at: datetime | None,
) -> Select[tuple[Claim]]:
    if predicate is not None:
        statement = statement.where(
            Claim.predicate_version_id
            == get_predicate_spec(predicate).predicate_version_id
        )
    if at is not None:
        # "True at this moment", which for an interval means started and not yet
        # ended, and for a point means stated at it. Half-open deliberately: a
        # candidacy ending on the 8th and another starting on the 8th is one
        # continuous run, not a day counted twice.
        statement = statement.where(
            (
                (Claim.valid_from <= at)
                & ((Claim.valid_to.is_(None)) | (Claim.valid_to > at))
            )
            | (Claim.valid_at == at)
        )
    return statement


def _rows(
    session: Session, statement: Select[tuple[Claim]], *, limit: int
) -> tuple[ClaimRow, ...]:
    """Run the query and resolve names and provenance in two more round trips.

    Batched rather than per row: a 100-claim page was 201 queries when this
    resolved names lazily, which is the shape that makes a list endpoint slow
    for reasons no single query explains.
    """

    claims = session.scalars(
        statement.order_by(
            func.coalesce(Claim.valid_from, Claim.valid_at).desc().nulls_last(),
            Claim.created_at.desc(),
        ).limit(limit)
    ).all()
    if not claims:
        return ()

    entity_ids = {claim.subject_id for claim in claims}
    entity_ids.update(claim.object_id for claim in claims if claim.object_id)
    entities = {
        entity.id: EntityRef(
            entity_id=entity.id, kind=entity.kind, name=entity.canonical_name
        )
        for entity in session.scalars(select(Entity).where(Entity.id.in_(entity_ids)))
    }

    provenance = _provenance(session, [claim.id for claim in claims])

    return tuple(
        ClaimRow(
            claim_id=claim.id,
            predicate=(
                slug := get_predicate_spec_by_id(claim.predicate_version_id).slug
            ),
            subject=entities[claim.subject_id],
            object=entities.get(claim.object_id) if claim.object_id else None,
            value=_value(slug, claim.value),
            raw_value=claim.value,
            valid_at=claim.valid_at,
            valid_at_precision=claim.valid_at_precision,
            valid_from=claim.valid_from,
            valid_from_precision=claim.valid_from_precision,
            valid_to=claim.valid_to,
            valid_to_precision=claim.valid_to_precision,
            assertion_count=provenance.get(claim.id, (0, ()))[0],
            asserted_by=provenance.get(claim.id, (0, ()))[1],
        )
        for claim in claims
    )


def _value(slug: str, raw: dict[str, Any] | None) -> ClaimValue | None:
    """Parse a payload, downgrading a contract violation to a warning.

    A payload that does not satisfy its predicate's schema is a real problem —
    `check_claim_ontology` exists to stop one being written — but discovering it
    while rendering a page is the wrong moment to raise. The row still comes
    back, with `raw_value` intact and `value` empty, so the damage is visible
    rather than fatal.
    """

    try:
        return parse_claim_value(slug, raw)
    except ValidationError:
        logger.warning("claim payload does not match the %s contract: %r", slug, raw)
        return None


def evidence_for(
    session: Session, claim_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, tuple[Evidence, ...]]:
    """What each claim cites, keyed by claim id.

    Separate from `claims_about`/`claims_with` and batched rather than per row,
    for two different reasons. Separate because a list of a hundred claims does
    not want a hundred excerpts, and joining them into the list query would make
    every page pay for what one panel needs. Batched because the alternative is
    the N+1 that every "show the source" feature is born with.

    This is the read surface's whole reason for existing on a project about
    *citable* facts. A claim you can see but not check is the thing the
    provenance model was built to prevent, and until this existed the module
    could render "Crowley won" without being able to show what said so.

    Ordered oldest first: the first assertion is the one that created the claim,
    and later ones are corroboration.
    """

    if not claim_ids:
        return {}

    rows = session.execute(
        select(
            ClaimAssertion.claim_id,
            ClaimAssertion.asserted_by,
            EvidenceAnchor.excerpt,
            Source.canonical_url,
            Source.title,
            SourceSnapshot.retrieved_at,
        )
        .join(EvidenceAnchor, EvidenceAnchor.id == ClaimAssertion.evidence_anchor_id)
        .join(SourceSnapshot, SourceSnapshot.id == EvidenceAnchor.source_snapshot_id)
        .join(Source, Source.id == SourceSnapshot.source_id)
        .where(ClaimAssertion.claim_id.in_(claim_ids))
        .order_by(ClaimAssertion.seq)
    )

    found: dict[uuid.UUID, tuple[Evidence, ...]] = {}
    for claim_id, asserted_by, excerpt, url, title, retrieved_at in rows:
        found[claim_id] = (
            *found.get(claim_id, ()),
            Evidence(
                asserted_by=asserted_by or "(unattributed)",
                excerpt=excerpt,
                source_url=url,
                source_title=title,
                retrieved_at=retrieved_at,
            ),
        )
    return found


def _provenance(
    session: Session, claim_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, tuple[int, tuple[str, ...]]]:
    """How many assertions each claim has, and who made them."""

    rows = session.execute(
        select(
            ClaimAssertion.claim_id,
            func.count(ClaimAssertion.id),
            func.array_agg(func.distinct(ClaimAssertion.asserted_by)),
        )
        .where(ClaimAssertion.claim_id.in_(claim_ids))
        .group_by(ClaimAssertion.claim_id)
    )
    return {
        claim_id: (count, tuple(sorted(writer for writer in writers if writer)))
        for claim_id, count, writers in rows
    }
