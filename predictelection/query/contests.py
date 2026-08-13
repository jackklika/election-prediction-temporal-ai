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
import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from predictelection.query.claims import ClaimRow, EntityRef, claims_with
from predictelection.query.entities import entity

# `query` reading `research` is the one edge here worth justifying. It is
# acyclic — `research` never imports `query` — and the alternative is a second
# copy of the key grammar, which is exactly the duplication that put
# `find_lookalikes` in `sql` earlier today. `ContestKey` is derived identity and
# would sit more naturally in `sql`, alongside `normalize_slug`; it lives in
# `research` with four sibling key types, and splitting one out of a cohesive
# module was not worth it for this.
from predictelection.research.contests import CONTEST_KEY_NAMESPACE, ContestKey
from predictelection.sql.base import TimePrecision
from predictelection.sql.entity import EntityIdentifier, resolve_entity
from predictelection.sql.predicate import ContestResultValue, ContestStage


logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class ContestDetail:
    """What a race *is* — the header a timeline page needs above the timeline.

    Read from the contest key rather than from claims, and that is the point.
    `contest_stage`, `contest_party`, `contest_for_office` and
    `contest_in_jurisdiction` are all seeded predicates with writers, and on this
    database every one of them has **zero** claims because the structure agent
    has not been re-run. Asking claims what a contest is therefore answers
    "nothing" for every contest in the graph.

    The key already carries all of it — division, office, cycle, stage, party —
    because it was designed so an importer and an agent could derive the same
    identity independently. Parsing it back turns identity into description for
    free, needs no agent run, and cannot disagree with the entity it names.

    Claims would still add what the key cannot say: the election date, the
    office's formal title, whether a runoff happened. When they exist, they
    refine this rather than replace it.
    """

    contest: EntityRef
    key: str
    division: str
    office: str
    cycle: int
    stage: ContestStage
    party: str | None = None
    jurisdiction: EntityRef | None = None
    """The division as an entity, when one is recorded under that OCD id — which
    is how a race links to the 47k jurisdictions and, eventually, to a polygon."""

    @property
    def label(self) -> str:
        return self.contest.name


def contest_detail(session: Session, contest_id: uuid.UUID) -> ContestDetail | None:
    """One contest described from its key, or None if it has no key.

    A contest with no `contest-key` identifier is one minted by name alone —
    possible, and worth returning None for rather than guessing, because every
    field here would otherwise be an invention.
    """

    contest = entity(session, contest_id)
    if contest is None:
        return None

    raw = session.scalar(
        select(EntityIdentifier.value).where(
            EntityIdentifier.entity_id == contest.entity_id,
            EntityIdentifier.namespace == CONTEST_KEY_NAMESPACE,
        )
    )
    if raw is None:
        return None
    return _detail(session, contest, raw)


def contests_in(
    session: Session,
    division_prefix: str,
    *,
    cycle: int | None = None,
    office: str | None = None,
    stage: ContestStage | None = None,
    limit: int = 200,
) -> tuple[ContestDetail, ...]:
    """Every contest under a division prefix — the race browser's list.

    Prefix matching on the key is what the key's shape was for: it begins with
    the OCD division, and OCD divisions nest, so
    `ocd-division/country:us/state:wi` selects Wisconsin's statewide races and
    every congressional district within it in one comparison.

    Filtering happens in Python after parsing rather than as more SQL `LIKE`s.
    The key's segments are ordered division/office/cycle/stage/party, so
    narrowing by cycle alone is not a prefix and would need a pattern with a
    wildcard in the middle — which no index helps with anyway. Parsing a few
    thousand keys is cheap and it cannot drift from `ContestKey`'s own grammar.

    Note this is a scan today: `uq_entity_identifier_value` is a plain btree, so
    a `LIKE 'prefix%'` cannot use it under a non-C collation. At 6,089 contest
    keys that is irrelevant. At a million it would want `text_pattern_ops`.
    """

    normalized = division_prefix.strip().lower()
    rows = session.execute(
        select(EntityIdentifier.entity_id, EntityIdentifier.value)
        .where(
            EntityIdentifier.namespace == CONTEST_KEY_NAMESPACE,
            EntityIdentifier.value.startswith(normalized),
        )
        .order_by(EntityIdentifier.value)
    ).all()

    found: list[ContestDetail] = []
    for entity_id, raw in rows:
        try:
            key = ContestKey.parse(raw)
        except ValueError:
            # A malformed key is a data problem, not a reason to fail a browse.
            logger.warning("skipping unparseable contest key %r", raw)
            continue
        if cycle is not None and key.cycle != cycle:
            continue
        if office is not None and key.office != office:
            continue
        if stage is not None and key.stage is not stage:
            continue
        reference = entity(session, entity_id)
        if reference is not None:
            found.append(_detail(session, reference, raw, key=key))
        if len(found) >= limit:
            break
    return tuple(found)


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
        if row.object is None or not (
            isinstance(row.value, ContestResultValue) and row.value.won
        ):
            continue
        office = office_of_contest.get(row.object.entity_id)
        if office is not None:
            pairs.append((office, _result(row)))
    return tuple(pairs)


# --------------------------------------------------------------------------


def _result(row: ClaimRow) -> ContestResultRow:
    """Narrowed on the predicate, which is the union's discriminator.

    A `contest_result` claim whose payload is not a `ContestResultValue` is a
    claim written against a contract it does not satisfy; the fields come back
    empty rather than the row vanishing, so the contradiction stays visible.
    """

    value = row.value if isinstance(row.value, ContestResultValue) else None
    assert row.object is not None
    return ContestResultRow(
        claim_id=row.claim_id,
        candidate=row.subject,
        contest=row.object,
        votes=value.votes if value else None,
        # Decimal all the way through: these are published shares, and adding
        # rounding error to a number someone can check against a source is
        # indefensible.
        share=value.share if value else None,
        place=value.place if value else None,
        won=value.won if value else None,
    )


def _detail(
    session: Session,
    contest: EntityRef,
    raw: str,
    *,
    key: ContestKey | None = None,
) -> ContestDetail:
    parsed = key or ContestKey.parse(raw)
    jurisdiction_id = session.scalar(
        select(EntityIdentifier.entity_id).where(
            EntityIdentifier.namespace == "ocd-division",
            EntityIdentifier.value == parsed.division,
        )
    )
    return ContestDetail(
        contest=contest,
        key=raw,
        division=parsed.division,
        office=parsed.office,
        cycle=parsed.cycle,
        stage=parsed.stage,
        party=parsed.party,
        jurisdiction=entity(session, jurisdiction_id) if jurisdiction_id else None,
    )
