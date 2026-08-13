"""Contests: who ran, who won, and which office it was for.

Two of the three functions here are thin wrappers over `claims_with`, which is
the point — `candidate_in` and `contest_result` are ordinary predicates and get
ordinary treatment. What they add is the shape a caller actually wants: a stint
with both ends and their precisions, and a result with its votes parsed out of
the value payload rather than left as a dict.

`winners_by_office` is the one that could not be a generic claim query. It walks
two predicates — `contest_result` for the outcome and `contest_for_office` for
the office — and that join is exactly the thing the data-model roadmap calls the
structural backbone: without it there is no way to ask which races share a
geography, which is what every crosstab is built on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from predictelection.query.claims import ClaimRow, EntityRef, claims_with
from predictelection.sql.entity import Entity, EntityIdentifier, resolve_entity
from predictelection.sql.base import TimePrecision


CONTEST_KEY_NAMESPACE = "contest-key"
"""Matches `research.contests.CONTEST_KEY_NAMESPACE`, restated rather than
imported: `query` sits beside `research`, not above it, and the string is the
stable part of that contract."""


@dataclass(frozen=True, slots=True)
class CandidateStint:
    """One continuous period a person was a candidate in one contest.

    A person can have several. A withdrawal ends a stint and a re-entry opens a
    new one — David Crowley has two for the 2026 Wisconsin primary — which is
    the whole reason this is not a `status` column.
    """

    claim_id: uuid.UUID
    """Kept so provenance stays reachable. A projected row that cannot be traced
    back to its claim cannot be cited, which on this project is the whole job —
    see `query.evidence_for`."""

    person: EntityRef
    contest: EntityRef
    started_at: datetime | None
    ended_at: datetime | None
    started_precision: TimePrecision | None = None
    ended_precision: TimePrecision | None = None

    @property
    def is_open(self) -> bool:
        return self.started_at is not None and self.ended_at is None


@dataclass(frozen=True, slots=True)
class ContestResultRow:
    """One candidate's result, with the payload unpacked."""

    claim_id: uuid.UUID
    candidate: EntityRef
    contest: EntityRef
    votes: int | None
    share: Decimal | None
    place: int | None
    won: bool | None
    """None means the source stated no outcome — not that they lost. A results
    table states counts; only a "Nominee" heading or a call states a winner."""


def contest_by_key(session: Session, contest_key: str) -> uuid.UUID | None:
    """The contest entity for a derived key, following any merge.

    Through `resolve_entity`, because a contest that was merged into another
    still owns its identifier row, and a caller asking by key would otherwise
    land on the duplicate.
    """

    entity_id = session.scalar(
        select(EntityIdentifier.entity_id).where(
            EntityIdentifier.namespace == CONTEST_KEY_NAMESPACE,
            EntityIdentifier.value == contest_key,
        )
    )
    return resolve_entity(session, entity_id) if entity_id else None


def candidates_in(
    session: Session,
    contest_id: uuid.UUID,
    *,
    at: datetime | None = None,
) -> tuple[CandidateStint, ...]:
    """Everyone who ran in this contest, as stints.

    With `at`, only those running at that moment — which is the query that
    proves the interval model works: three dates over one primary return three
    different fields, from claims alone, with nothing overwritten.
    """

    return tuple(
        CandidateStint(
            claim_id=row.claim_id,
            person=row.subject,
            contest=row.object or row.subject,
            started_at=row.valid_from,
            ended_at=row.valid_to,
            started_precision=row.valid_from_precision,
            ended_precision=row.valid_to_precision,
        )
        for row in claims_with(
            session, "candidate_in", object_ids=[contest_id], at=at, limit=500
        )
        if row.object is not None
    )


def results_for(
    session: Session, contest_id: uuid.UUID
) -> tuple[ContestResultRow, ...]:
    """The result of one contest, best-placed first.

    Ordered by votes rather than by `place`, because `place` is often absent —
    a results table prints counts and leaves the ranking implied.
    """

    rows = [
        _result(row)
        for row in claims_with(
            session, "contest_result", object_ids=[contest_id], limit=500
        )
        if row.object is not None
    ]
    return tuple(sorted(rows, key=lambda row: (row.votes is None, -(row.votes or 0))))


def winners_by_office(
    session: Session,
    *,
    office_id: uuid.UUID | None = None,
    limit: int = 200,
) -> tuple[tuple[EntityRef, ContestResultRow], ...]:
    """(office, winning result) pairs — the join backtesting starts from.

    Only returns results whose source actually stated a win. `won` is nullable
    on purpose and `False` would be an assertion nobody made, so a contest whose
    results were imported from a vote table without a "Nominee" heading is
    absent here rather than wrongly attributed. That is a real gap in the data
    today, not in this query: general-election results carry `won = NULL`.
    """

    links = claims_with(session, "contest_for_office", limit=1000)
    if office_id is not None:
        links = tuple(
            link for link in links if link.object and link.object.entity_id == office_id
        )
    # contest -> office. `contest_for_office` points contest at office, and a
    # result points candidate at contest, so the contest id is the hinge.
    office_of_contest = {
        link.subject.entity_id: link.object for link in links if link.object is not None
    }
    if not office_of_contest:
        return ()

    pairs: list[tuple[EntityRef, ContestResultRow]] = []
    for row in claims_with(
        session, "contest_result", object_ids=list(office_of_contest), limit=limit
    ):
        if row.object is None or not (row.value or {}).get("won"):
            continue
        office = office_of_contest.get(row.object.entity_id)
        if office is not None:
            pairs.append((office, _result(row)))
    return tuple(pairs)


# --------------------------------------------------------------------------


def _result(row: ClaimRow) -> ContestResultRow:
    value = row.value or {}
    share = value.get("share")
    assert row.object is not None
    return ContestResultRow(
        claim_id=row.claim_id,
        candidate=row.subject,
        contest=row.object,
        votes=value.get("votes"),
        # Decimal, not float: these are published shares and adding rounding
        # error to a number someone can check against a source is indefensible.
        share=Decimal(str(share)) if share is not None else None,
        place=value.get("place"),
        won=value.get("won"),
    )


def entity(session: Session, entity_id: uuid.UUID) -> EntityRef | None:
    """One entity as a reference, for a caller holding only an id."""

    found = session.get(Entity, resolve_entity(session, entity_id))
    if found is None:
        return None
    return EntityRef(entity_id=found.id, kind=found.kind, name=found.canonical_name)
