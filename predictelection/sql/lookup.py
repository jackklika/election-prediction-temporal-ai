"""Read-only views of the graph, for an agent to link against.

The classic knowledge-graph construction pipeline discovers and links entities
*before* extracting relations. Ours does the reverse — the agent reports entities
and relations together and we resolve afterwards — and the cost is measurable:
two runs on one subject produced 11 event entities for 6 real debates, because
the agent re-described each debate instead of re-using what it had already
written. That is the classic "does not distinguish entities from different
expressions, which prevents knowledge aggregation" failure.

These functions let the agent look first. They return the *canonical name* an
entity already has, so the model can echo it back rather than invent a new
phrasing, which prevents the fork rather than cleaning up after it.

Two properties are load-bearing and easy to lose:

- **Truncation is reported, never silent.** A capped list reads to a model as
  "nothing else exists", so it re-describes what it cannot see. Every result
  carries `truncated`, and callers are expected to say so in the prompt.
- **Ordering matches the question.** Events come back most-recent-first, because
  the rows a cap drops have to be the least relevant ones. Ordering events
  alphabetically means the cap keeps whatever starts with "A".

Read-only by construction: nothing here writes, so a lookup is safe to repeat
and cannot corrupt a run that is retried.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
import uuid

from sqlalchemy import ColumnElement, Select, func, nulls_last, select
from sqlalchemy.orm import Session

from predictelection.sql.claim import Claim
from predictelection.sql.entity import (
    Entity,
    EntityAlias,
    EntityKind,
    normalize_entity_name,
)
from predictelection.sql.predicate import PoliticalEventKind, get_predicate_spec


DEFAULT_LIMIT = 20


@dataclass(frozen=True, slots=True)
class EntityMatch:
    """What the agent needs to decide "is this the thing I am looking at?"."""

    entity_id: uuid.UUID
    kind: EntityKind
    canonical_name: str
    aliases: tuple[str, ...] = ()
    occurred_at: datetime | None = None
    """For events, the date from event_occurrence — the strongest disambiguator."""


@dataclass(frozen=True, slots=True)
class EntityMatches:
    """Matches plus whether the limit hid any.

    `truncated` is the whole reason this is not a bare list. A caller that
    renders a capped list without saying it was capped is telling the model the
    graph is empty beyond that point, which is exactly how duplicates get made.

    Behaves like a sequence so call sites can keep iterating, indexing and
    calling len() on it.
    """

    matches: tuple[EntityMatch, ...] = ()
    truncated: bool = False

    def __iter__(self) -> Iterator[EntityMatch]:
        return iter(self.matches)

    def __len__(self) -> int:
        return len(self.matches)

    def __getitem__(self, index: int) -> EntityMatch:
        return self.matches[index]

    def __bool__(self) -> bool:
        return bool(self.matches)


def find_entities(
    session: Session,
    *,
    name: str | None = None,
    kind: EntityKind | None = None,
    limit: int = DEFAULT_LIMIT,
) -> EntityMatches:
    """Entities already in the graph, by name fragment and kind.

    Matches on the normalized alias key and on a substring of the canonical
    name, so a partial or differently-cased query still finds the entity the
    agent should reuse.
    """

    statement = select(Entity, _latest_occurrence())
    if kind is not None:
        statement = statement.where(Entity.kind == kind)
    if name:
        statement = statement.where(_name_matches(name))
    return _materialize(session, statement.order_by(Entity.canonical_name), limit=limit)


def find_events(
    session: Session,
    *,
    name: str | None = None,
    jurisdiction_id: uuid.UUID | None = None,
    participant_ids: Sequence[uuid.UUID] | None = None,
    event_kind: PoliticalEventKind | None = None,
    occurred_after: datetime | None = None,
    occurred_before: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> EntityMatches:
    """Events, narrowed by who was there, what kind they are, when and where.

    Date is the disambiguator that name similarity cannot supply: two debates a
    month apart share almost all of their words, while the same debate described
    twice shares a date.

    `participant_ids` is what makes the result about the subject being researched
    rather than about the graph as a whole. Without it a cap returns whatever
    sorts first across every event ever recorded, which tells the agent nothing
    about the person it was asked to look up. Plural because a name can resolve
    to several people, and showing all of their events beats picking one at
    random and calling it the subject.

    `event_kind` matters as soon as a second kind of event exists: rallies and
    town halls are EVENT entities too, and an unfiltered list invites the agent
    to treat them as debates it already reported.
    """

    occurred_at = _latest_occurrence()
    statement = select(Entity, occurred_at).where(Entity.kind == EntityKind.EVENT)

    if name:
        statement = statement.where(_name_matches(name))

    if occurred_after is not None or occurred_before is not None:
        # "has an occurrence claim in this window" rather than "its latest one is
        # in this window": a postponement is a second claim over a new interval,
        # and an event should still be findable under the date first announced.
        moment = func.coalesce(Claim.valid_from, Claim.valid_at)
        dated = select(Claim.subject_id).where(
            Claim.predicate_version_id == _version_of("event_occurrence")
        )
        if occurred_after is not None:
            dated = dated.where(moment >= occurred_after)
        if occurred_before is not None:
            dated = dated.where(moment <= occurred_before)
        statement = statement.where(Entity.id.in_(dated))

    if event_kind is not None:
        of_kind = select(Claim.subject_id).where(
            Claim.predicate_version_id == _version_of("event_kind"),
            Claim.value["kind"].astext == event_kind.value,
        )
        statement = statement.where(Entity.id.in_(of_kind))

    if participant_ids:
        # participated_in points person -> event, so the event ids are the objects.
        attended = select(Claim.object_id).where(
            Claim.predicate_version_id == _version_of("participated_in"),
            Claim.subject_id.in_(participant_ids),
        )
        statement = statement.where(Entity.id.in_(attended))

    if jurisdiction_id is not None:
        located = select(Claim.subject_id).where(
            Claim.predicate_version_id == _version_of("event_in_jurisdiction"),
            Claim.object_id == jurisdiction_id,
        )
        statement = statement.where(Entity.id.in_(located))

    # Most recent first: whatever the limit drops should be the least relevant
    # thing, and undated events are the least useful of all.
    statement = statement.order_by(
        nulls_last(occurred_at.desc()), Entity.canonical_name
    )
    return _materialize(session, statement, limit=limit)


# --------------------------------------------------------------------------


def _version_of(slug: str) -> uuid.UUID:
    return get_predicate_spec(slug).predicate_version_id


def _name_matches(name: str) -> ColumnElement[bool]:
    """Canonical name substring OR normalized alias, for any entity kind.

    Events get the alias branch too. They did not before, so an event recorded
    under one title and later renamed was unfindable under the name the agent
    had already used for it — a fork with no way back.
    """

    by_alias = select(EntityAlias.entity_id).where(
        EntityAlias.normalized_name.contains(normalize_entity_name(name))
    )
    return Entity.canonical_name.ilike(f"%{name.strip()}%") | Entity.id.in_(by_alias)


def _latest_occurrence() -> ColumnElement[datetime]:
    """When this entity's most recently recorded event_occurrence claim puts it.

    Annotated non-optional because SQLAlchemy's column types do not model
    nullability; it yields NULL, and therefore None, for anything that is not a
    dated event.

    Correlated on Entity, so it selects as a column and orders without a join.
    Ordered by
    created_at rather than by the moment itself because a postponement is a new
    claim: the last one written is the current belief, even when it moves the
    event earlier.
    """

    return (
        select(func.coalesce(Claim.valid_from, Claim.valid_at))
        .where(
            Claim.subject_id == Entity.id,
            Claim.predicate_version_id == _version_of("event_occurrence"),
        )
        .order_by(Claim.created_at.desc())
        .limit(1)
        .correlate(Entity)
        .scalar_subquery()
    )


def _materialize(
    session: Session,
    statement: Select[tuple[Entity, datetime]],
    *,
    limit: int,
) -> EntityMatches:
    """Run the query, detect truncation, and load aliases in one extra round trip.

    Asks for limit + 1 rows purely to answer "was there more?". Aliases used to
    cost a query per row, plus another for the occurrence date; both are batched
    here, so a 50-row context block is 2 queries rather than 101.
    """

    rows = session.execute(statement.limit(limit + 1)).all()
    truncated = len(rows) > limit
    rows = rows[:limit]

    aliases = _aliases_for(session, [entity.id for entity, _ in rows])
    return EntityMatches(
        matches=tuple(
            EntityMatch(
                entity_id=entity.id,
                kind=entity.kind,
                canonical_name=entity.canonical_name,
                aliases=aliases.get(entity.id, ()),
                occurred_at=occurred_at,
            )
            for entity, occurred_at in rows
        ),
        truncated=truncated,
    )


def _aliases_for(
    session: Session, entity_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, tuple[str, ...]]:
    if not entity_ids:
        return {}
    grouped: dict[uuid.UUID, tuple[str, ...]] = {}
    rows = session.execute(
        select(EntityAlias.entity_id, EntityAlias.name)
        .where(EntityAlias.entity_id.in_(entity_ids))
        .order_by(EntityAlias.name)
    )
    for entity_id, alias in rows:
        grouped[entity_id] = (*grouped.get(entity_id, ()), alias)
    return grouped
