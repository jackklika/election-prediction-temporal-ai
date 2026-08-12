"""Turning a scraped name into a stable Entity identity.

Scrapers observe strings; claims need entity IDs. This is the bridge, and it is
the most-used operation in any ingestion path.

The design principle is that resolution does not have to be right the first
time, it has to be *correctable*. EntityRedirect already provides an append-only,
audited merge mechanism, so a wrong split can always be repaired later. That lets
this step be total, deterministic, and fast, and push every genuinely uncertain
judgment into a queue rather than guessing.

Determinism matters concretely for Temporal: an activity that asked an LLM to
adjudicate a merge could answer differently on attempt one and attempt two, and
create two entities for one thing. Every tier here is a plain database lookup.

Tiers, cheapest first:

0. External identifier. Exact, and always wins.
1. Exact normalized alias within the same kind.
2. Create a new entity.

Fuzzy candidate generation belongs above tier 2 and is deliberately absent: you
cannot tune a matcher before you have duplicates to tune it against. When it
arrives it should propose merges for review rather than apply them inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from predictelection.sql.entity import (
    Entity,
    EntityAlias,
    EntityIdentifier,
    EntityKind,
    new_entity_alias,
    normalize_entity_name,
    normalize_identifier_namespace,
    resolve_entity,
)


class ResolutionMethod(StrEnum):
    IDENTIFIER = "identifier"
    """Matched an external identifier, which is exact."""

    ALIAS = "alias"
    """Matched one existing entity of the same kind by normalized name."""

    CREATED = "created"
    """No confident match, so a new entity was minted."""


@dataclass(frozen=True, slots=True)
class ExternalIdentifier:
    namespace: str
    value: str

    def normalized(self) -> ExternalIdentifier:
        return ExternalIdentifier(
            namespace=normalize_identifier_namespace(self.namespace),
            value=self.value.strip(),
        )


@dataclass(frozen=True, slots=True)
class EntityMention:
    """A named thing as a scraper observed it, before it has an identity.

    This is the type domain models should produce. Carry identifiers whenever the
    source offers them: a Wikidata QID or an FEC ID resolves exactly, while a name
    only ever resolves heuristically.
    """

    kind: EntityKind
    name: str
    identifiers: tuple[ExternalIdentifier, ...] = ()
    aliases: tuple[str, ...] = field(default=())

    identifiers_are_authoritative: bool = False
    """Whether an identifier miss should mint rather than fall back to the name.

    Off by default, because for most identifiers the name is still evidence: the
    OCD import saying "Michigan" with an `ocd-division` ID should attach that ID
    to the Michigan a debate already created, not mint a second one.

    On for a *derived* key, where the identifier is not a fact about the entity
    but the definition of it. Two debates titled "Michigan Senate Debate" on
    different days are different events; with this off they merge on the name
    and the survivor ends up carrying both keys, which is worse than either
    outcome on its own.
    """

    def __post_init__(self) -> None:
        normalize_entity_name(self.name)  # reject blank or whitespace-only early


@dataclass(frozen=True, slots=True)
class Resolution:
    entity: Entity
    method: ResolutionMethod
    ambiguous_with: tuple[uuid.UUID, ...] = ()
    """Other same-kind entities sharing this normalized name.

    Non-empty means the name was not decisive and a new entity was created
    anyway. These are the pairs a future merge-proposal step should look at.
    """

    @property
    def entity_id(self) -> uuid.UUID:
        return self.entity.id

    @property
    def created(self) -> bool:
        return self.method is ResolutionMethod.CREATED


def _kinds_compatible(stored: EntityKind, observed: EntityKind) -> bool:
    """OTHER is the unknown kind, so it is compatible with anything."""

    return (
        stored is observed or stored is EntityKind.OTHER or observed is EntityKind.OTHER
    )


def _match_by_identifier(
    session: Session,
    identifiers: tuple[ExternalIdentifier, ...],
    *,
    kind: EntityKind,
) -> Entity | None:
    for identifier in identifiers:
        normalized = identifier.normalized()
        entity_id = session.scalar(
            select(EntityIdentifier.entity_id).where(
                EntityIdentifier.namespace == normalized.namespace,
                EntityIdentifier.value == normalized.value,
            )
        )
        if entity_id is None:
            continue

        entity = session.get(Entity, resolve_entity(session, entity_id))
        assert entity is not None
        if not _kinds_compatible(entity.kind, kind):
            # An exact key pointing at an incompatible kind is a contradiction,
            # not the fuzzy uncertainty an ambiguous name represents. Two things
            # cannot share a Wikidata QID, so a scraper is emitting bad data and
            # should hear about it now rather than through a misaligned claim.
            raise ValueError(
                f"{normalized.namespace}:{normalized.value} already identifies "
                f"{entity.kind} {entity.id}, not a {kind}"
            )
        return entity
    return None


def _match_by_alias(
    session: Session,
    *,
    kind: EntityKind,
    normalized_name: str,
) -> tuple[Entity | None, tuple[uuid.UUID, ...]]:
    """Entities of a compatible kind already known by this normalized name.

    uq_entity_alias_identity is scoped per entity, so one normalized name can
    legitimately belong to several entities — two different people called John
    Smith. More than one hit is therefore ambiguity, not a match.

    OTHER is treated as compatible in both directions: it is the kind a scraper
    falls back to when it could not tell, so matching it lets a later, better
    typed mention claim the same identity instead of forking a duplicate.
    """

    candidate_ids = list(
        session.scalars(
            select(Entity.id)
            .join(EntityAlias, EntityAlias.entity_id == Entity.id)
            .where(
                EntityAlias.normalized_name == normalized_name,
                Entity.kind.in_({kind, EntityKind.OTHER}),
            )
            .distinct()
        )
    )
    resolved = {resolve_entity(session, candidate) for candidate in candidate_ids}
    if len(resolved) != 1:
        return None, tuple(sorted(resolved))
    entity = session.get(Entity, resolved.pop())
    return entity, ()


def _record_alias(session: Session, entity: Entity, name: str) -> None:
    """Keep the observed surface form, so the match index grows as we read.

    EntityAlias is immutable and uniquely keyed, so re-observing a known spelling
    must be a no-op rather than an insert.
    """

    normalized_name = normalize_entity_name(name)
    already_known = session.scalar(
        select(EntityAlias.id).where(
            EntityAlias.entity_id == entity.id,
            EntityAlias.normalized_name == normalized_name,
            EntityAlias.language.is_(None),
        )
    )
    if already_known is None:
        session.add(new_entity_alias(entity_id=entity.id, name=name))


def _record_identifiers(
    session: Session,
    entity: Entity,
    identifiers: tuple[ExternalIdentifier, ...],
) -> None:
    for identifier in identifiers:
        normalized = identifier.normalized()
        owner = session.scalar(
            select(EntityIdentifier.entity_id).where(
                EntityIdentifier.namespace == normalized.namespace,
                EntityIdentifier.value == normalized.value,
            )
        )
        if owner is None:
            session.add(
                EntityIdentifier(
                    entity_id=entity.id,
                    namespace=normalized.namespace,
                    value=normalized.value,
                )
            )
        elif resolve_entity(session, owner) != entity.id:
            raise ValueError(
                f"{normalized.namespace}:{normalized.value} already identifies "
                f"{owner}, not {entity.id}"
            )


def resolve_entity_mention(session: Session, mention: EntityMention) -> Resolution:
    """Resolve a scraped mention to an entity, creating one if needed.

    Always returns an entity, and never raises on an unrecognised name — an
    unknown person is a new person, not an error. Flushes so the caller can use
    the ID immediately.
    """

    normalized_name = normalize_entity_name(mention.name)

    entity = _match_by_identifier(session, mention.identifiers, kind=mention.kind)
    method = ResolutionMethod.IDENTIFIER
    ambiguous_with: tuple[uuid.UUID, ...] = ()

    # A miss on an authoritative identifier means this is a new thing, not a
    # thing to look up by name. Falling through would merge two entities the
    # keys say are distinct, and then record both keys on the survivor.
    name_may_decide = not (
        mention.identifiers and mention.identifiers_are_authoritative
    )

    if entity is None and name_may_decide:
        entity, ambiguous_with = _match_by_alias(
            session, kind=mention.kind, normalized_name=normalized_name
        )
        method = ResolutionMethod.ALIAS

    if entity is None:
        entity = Entity(kind=mention.kind, canonical_name=mention.name)
        session.add(entity)
        session.flush()
        method = ResolutionMethod.CREATED
    elif entity.kind is EntityKind.OTHER and mention.kind is not EntityKind.OTHER:
        # A better-typed mention refines a placeholder rather than forking one.
        entity.kind = mention.kind

    _record_alias(session, entity, mention.name)
    for alias in mention.aliases:
        _record_alias(session, entity, alias)
    _record_identifiers(session, entity, mention.identifiers)
    session.flush()

    return Resolution(entity=entity, method=method, ambiguous_with=ambiguous_with)
