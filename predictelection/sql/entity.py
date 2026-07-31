from __future__ import annotations

from enum import StrEnum
import unicodedata
import uuid

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    event,
    literal,
    select,
)
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from predictelection.sql.base import (
    Base,
    Immutable,
    created_at_timestamp,
    enum_type,
    uuid_primary_key,
)


MAX_REDIRECT_DEPTH = 16
"""How far resolve_entity will follow redirects before calling it a cycle."""


def normalize_entity_name(name: str) -> str:
    """Fold a display name into the key used for alias matching.

    Deliberately conservative: Unicode form, case, and whitespace only. Fuzzy
    matching on punctuation, initials, or nicknames is a retrieval concern that
    belongs in the resolution step, not in a stored equality key.
    """

    folded = unicodedata.normalize("NFKC", name).casefold()
    normalized = " ".join(folded.split())
    if not normalized:
        raise ValueError("entity names must contain more than whitespace")
    return normalized


def normalize_identifier_namespace(namespace: str) -> str:
    """Fold an external namespace to the form ck_..._namespace_normalized wants."""

    normalized = unicodedata.normalize("NFKC", namespace).strip().lower()
    if not normalized:
        raise ValueError("identifier namespaces must contain more than whitespace")
    return normalized


class EntityKind(StrEnum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PARTY = "party"
    """Distinct from ORGANIZATION: party is load-bearing for elections.

    Primaries belong to one, affiliation changes over time, and results are read
    by party. Leaving it inside ORGANIZATION would make every one of those a
    free-text guess.
    """

    JURISDICTION = "jurisdiction"
    OFFICE = "office"
    ELECTION = "election"
    CONTEST = "contest"
    EVENT = "event"
    MARKET = "market"
    OPTION = "option"
    OTHER = "other"


class Entity(Base):
    """A stable identity for a named thing.

    Names may improve over time. Identity changes use EntityRedirect so old
    foreign keys and external references continue to resolve.
    """

    __tablename__ = "entity"

    id: Mapped[uuid_primary_key]
    kind: Mapped[EntityKind] = mapped_column(enum_type(EntityKind, name="entity_kind"))
    canonical_name: Mapped[str] = mapped_column(Text)
    created_at: Mapped[created_at_timestamp]

    identifiers: Mapped[list[EntityIdentifier]] = relationship(
        back_populates="entity",
        passive_deletes="all",
    )
    aliases: Mapped[list[EntityAlias]] = relationship(
        back_populates="entity",
        passive_deletes="all",
    )
    redirect: Mapped[EntityRedirect | None] = relationship(
        foreign_keys="EntityRedirect.duplicate_entity_id",
        back_populates="duplicate_entity",
        uselist=False,
        passive_deletes="all",
    )

    __table_args__ = (Index("ix_entity_kind_canonical_name", "kind", "canonical_name"),)


class EntityIdentifier(Immutable, Base):
    """An immutable identifier assigned by an external namespace.

    Many per entity by design — Wikidata and OCD and FEC coexist rather than one
    winning. `namespace` is a foreign key into the registry rather than free
    text, so a typo cannot invent a scheme that silently matches nothing, and a
    scheme can be deprecated without touching these rows.
    """

    __tablename__ = "entity_identifier"

    id: Mapped[uuid_primary_key]
    entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    namespace: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("identifier_namespace.namespace", ondelete="RESTRICT"),
    )
    value: Mapped[str] = mapped_column(Text)
    research_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("research_run.id", ondelete="RESTRICT")
    )
    """Which run asserted this. Everything else here is citable; so is this."""

    created_at: Mapped[created_at_timestamp]

    entity: Mapped[Entity] = relationship(back_populates="identifiers")

    __table_args__ = (
        UniqueConstraint(
            "namespace",
            "value",
            name="uq_entity_identifier_namespace_value",
        ),
        Index("ix_entity_identifier_entity_id", "entity_id"),
        Index("ix_entity_identifier_research_run_id", "research_run_id"),
    )


class EntityAlias(Immutable, Base):
    """An immutable alternate name used during entity resolution."""

    __tablename__ = "entity_alias"

    id: Mapped[uuid_primary_key]
    entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    name: Mapped[str] = mapped_column(Text)
    normalized_name: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[created_at_timestamp]

    entity: Mapped[Entity] = relationship(back_populates="aliases")

    __table_args__ = (
        UniqueConstraint(
            "entity_id",
            "normalized_name",
            "language",
            name="uq_entity_alias_identity",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint(
            "name <> '' AND normalized_name <> ''",
            name="names_nonempty",
        ),
        Index("ix_entity_alias_normalized_name", "normalized_name"),
    )


class EntityRedirect(Immutable, Base):
    """An append-only declaration that a duplicate resolves to another entity."""

    __tablename__ = "entity_redirect"

    duplicate_entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    canonical_entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    reason: Mapped[str] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[created_at_timestamp]

    duplicate_entity: Mapped[Entity] = relationship(
        foreign_keys=[duplicate_entity_id],
        back_populates="redirect",
    )
    canonical_entity: Mapped[Entity] = relationship(foreign_keys=[canonical_entity_id])

    __table_args__ = (
        CheckConstraint(
            "duplicate_entity_id <> canonical_entity_id",
            name="different_entities",
        ),
        CheckConstraint("reason <> '' AND created_by <> ''", name="audit_nonempty"),
        Index("ix_entity_redirect_canonical_entity_id", "canonical_entity_id"),
    )


@event.listens_for(EntityAlias, "before_insert")
def _derive_alias_normalized_name(
    mapper: object,
    connection: object,
    alias: EntityAlias,
) -> None:
    del mapper, connection
    alias.normalized_name = normalize_entity_name(alias.name)


@event.listens_for(EntityIdentifier, "before_insert")
def _normalize_identifier_namespace(
    mapper: object,
    connection: object,
    identifier: EntityIdentifier,
) -> None:
    del mapper, connection
    identifier.namespace = normalize_identifier_namespace(identifier.namespace)


def new_entity_alias(
    *,
    entity_id: uuid.UUID,
    name: str,
    language: str | None = None,
) -> EntityAlias:
    """Build an alias with its matching key already derived."""

    return EntityAlias(
        entity_id=entity_id,
        name=name,
        normalized_name=normalize_entity_name(name),
        language=language,
    )


def resolve_entity(
    session: Session,
    entity_id: uuid.UUID,
    *,
    max_depth: int = MAX_REDIRECT_DEPTH,
) -> uuid.UUID:
    """Follow EntityRedirect hops and return the canonical entity ID.

    A single hop is not enough: nothing in the schema stops A redirecting to B
    while B redirects to C, so a one-hop read would hand back a duplicate. The
    walk uses UNION ALL rather than UNION so that a cycle repeats a visited ID
    instead of silently terminating.
    """

    chain = select(
        literal(entity_id, Uuid).label("entity_id"),
        literal(0).label("depth"),
    ).cte("redirect_chain", recursive=True)
    chain = chain.union_all(
        select(
            EntityRedirect.canonical_entity_id.label("entity_id"),
            (chain.c.depth + 1).label("depth"),
        )
        .join(chain, EntityRedirect.duplicate_entity_id == chain.c.entity_id)
        .where(chain.c.depth < max_depth)
    )
    visited = list(
        session.execute(select(chain.c.entity_id).order_by(chain.c.depth)).scalars()
    )

    seen: set[uuid.UUID] = set()
    for candidate in visited:
        if candidate in seen:
            raise ValueError(f"entity redirect cycle reached {candidate}")
        seen.add(candidate)

    terminal = visited[-1]
    if len(visited) > max_depth:
        still_redirected = session.scalar(
            select(EntityRedirect.canonical_entity_id).where(
                EntityRedirect.duplicate_entity_id == terminal
            )
        )
        if still_redirected is not None:
            raise ValueError(
                f"entity redirect chain from {entity_id} exceeds {max_depth} hops"
            )
    return terminal
