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

Read-only by construction: nothing here writes, so a lookup is safe to repeat
and cannot corrupt a run that is retried.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from predictelection.sql.claim import Claim
from predictelection.sql.entity import (
    Entity,
    EntityAlias,
    EntityKind,
    normalize_entity_name,
)
from predictelection.sql.predicate import get_predicate_spec


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


def find_entities(
    session: Session,
    *,
    name: str | None = None,
    kind: EntityKind | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[EntityMatch]:
    """Entities already in the graph, by name fragment and kind.

    Matches on the normalized alias key and on a substring of the canonical
    name, so a partial or differently-cased query still finds the entity the
    agent should reuse.
    """

    statement = select(Entity)
    if kind is not None:
        statement = statement.where(Entity.kind == kind)
    if name:
        normalized = normalize_entity_name(name)
        by_alias = select(EntityAlias.entity_id).where(
            EntityAlias.normalized_name.contains(normalized)
        )
        statement = statement.where(
            Entity.canonical_name.ilike(f"%{name.strip()}%") | Entity.id.in_(by_alias)
        )
    statement = statement.order_by(Entity.canonical_name).limit(limit)
    return [_match(session, entity) for entity in session.scalars(statement)]


def find_events(
    session: Session,
    *,
    name: str | None = None,
    jurisdiction_id: uuid.UUID | None = None,
    occurred_after: datetime | None = None,
    occurred_before: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[EntityMatch]:
    """Events, optionally narrowed by when they happened and where.

    Date is the disambiguator that name similarity cannot supply: two debates a
    month apart share almost all of their words, while the same debate described
    twice shares a date. An agent given the date range it is writing about will
    find the existing entity even when it would have phrased the title
    differently.
    """

    statement = select(Entity).where(Entity.kind == EntityKind.EVENT)
    if name:
        statement = statement.where(Entity.canonical_name.ilike(f"%{name.strip()}%"))

    if occurred_after is not None or occurred_before is not None:
        occurrence = get_predicate_spec("event_occurrence").predicate_version_id
        dated = select(Claim.subject_id).where(Claim.predicate_version_id == occurrence)
        if occurred_after is not None:
            dated = dated.where(Claim.valid_from >= occurred_after)
        if occurred_before is not None:
            dated = dated.where(Claim.valid_from <= occurred_before)
        statement = statement.where(Entity.id.in_(dated))

    if jurisdiction_id is not None:
        located = select(Claim.subject_id).where(
            Claim.predicate_version_id
            == get_predicate_spec("event_in_jurisdiction").predicate_version_id,
            Claim.object_id == jurisdiction_id,
        )
        statement = statement.where(Entity.id.in_(located))

    statement = statement.order_by(Entity.canonical_name).limit(limit)
    return [_match(session, entity) for entity in session.scalars(statement)]


def _match(session: Session, entity: Entity) -> EntityMatch:
    aliases = tuple(
        session.scalars(
            select(EntityAlias.name).where(EntityAlias.entity_id == entity.id)
        )
    )
    return EntityMatch(
        entity_id=entity.id,
        kind=entity.kind,
        canonical_name=entity.canonical_name,
        aliases=aliases,
        occurred_at=_occurred_at(session, entity),
    )


def _occurred_at(session: Session, entity: Entity) -> datetime | None:
    if entity.kind is not EntityKind.EVENT:
        return None
    return session.scalar(
        select(Claim.valid_from).where(
            Claim.subject_id == entity.id,
            Claim.predicate_version_id
            == get_predicate_spec("event_occurrence").predicate_version_id,
        )
    )
