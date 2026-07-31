from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any, Mapping
import uuid

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
)
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from predictelection.sql.base import (
    Base,
    Immutable,
    canonical_json_sha256,
    created_at_timestamp,
    enum_type,
    nullable_jsonb,
)
from predictelection.sql.entity import EntityKind


_PREDICATE_ID_NAMESPACE = uuid.UUID("6d453df2-c665-45f7-bc5b-3d8140cc75da")
_PREDICATE_VERSION_ID_NAMESPACE = uuid.UUID("43157bf4-86e2-484f-b36a-bbde3717d533")


class PredicateTarget(StrEnum):
    ENTITY = "entity"
    VALUE = "value"


class TemporalMode(StrEnum):
    TIMELESS = "timeless"
    OPTIONAL = "optional"
    REQUIRED = "required"


class PoliticalEventKind(StrEnum):
    DEBATE = "debate"
    FORUM = "forum"
    TOWN_HALL = "town_hall"
    SPEECH = "speech"
    INTERVIEW = "interview"
    ELECTION = "election"
    OTHER = "other"


class PredicateValue(BaseModel):
    """Base for the JSON value of one literal predicate."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EventKindValue(PredicateValue):
    kind: PoliticalEventKind


class PublicStatementValue(PredicateValue):
    """A normalized position; the verbatim passage belongs in EvidenceAnchor."""

    topic: str = Field(min_length=1, max_length=500)
    position: str = Field(min_length=1, max_length=2_000)
    summary: str | None = Field(default=None, max_length=5_000)


class Predicate(Base):
    """Stable identity and human-readable metadata for a predicate."""

    __tablename__ = "predicate"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(100))
    label: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    created_at: Mapped[created_at_timestamp]

    versions: Mapped[list[PredicateVersion]] = relationship(back_populates="predicate")

    __table_args__ = (
        UniqueConstraint("slug", name="uq_predicate_slug"),
        CheckConstraint(
            "slug ~ '^[a-z][a-z0-9_]*$'",
            name="slug_normalized",
        ),
    )


class PredicateVersion(Immutable, Base):
    """An immutable contract for claims using a predicate."""

    __tablename__ = "predicate_version"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    predicate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("predicate.id", ondelete="RESTRICT")
    )
    version: Mapped[int] = mapped_column(Integer)
    target_kind: Mapped[PredicateTarget] = mapped_column(
        enum_type(PredicateTarget, name="predicate_target")
    )
    temporal_mode: Mapped[TemporalMode] = mapped_column(
        enum_type(TemporalMode, name="predicate_temporal_mode")
    )
    value_model_path: Mapped[str | None] = mapped_column(String(500))
    value_schema: Mapped[dict[str, Any] | None] = mapped_column(nullable_jsonb())
    schema_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[created_at_timestamp]

    predicate: Mapped[Predicate] = relationship(back_populates="versions")
    subject_kinds: Mapped[list[PredicateSubjectKind]] = relationship(
        back_populates="predicate_version"
    )
    object_kinds: Mapped[list[PredicateObjectKind]] = relationship(
        back_populates="predicate_version"
    )

    __table_args__ = (
        UniqueConstraint(
            "predicate_id",
            "version",
            name="uq_predicate_version_number",
        ),
        UniqueConstraint(
            "id",
            "target_kind",
            name="uq_predicate_version_id_target",
        ),
        CheckConstraint("version > 0", name="version_positive"),
        CheckConstraint(
            """
            (
                target_kind = 'entity'
                AND value_model_path IS NULL
                AND value_schema IS NULL
            )
            OR
            (
                target_kind = 'value'
                AND value_model_path IS NOT NULL
                AND value_schema IS NOT NULL
            )
            """,
            name="value_contract_matches_target",
        ),
        CheckConstraint(
            "schema_hash ~ '^[0-9a-f]{64}$'",
            name="schema_hash_lowercase_hex",
        ),
    )


class PredicateSubjectKind(Immutable, Base):
    """An entity kind allowed in the subject position."""

    __tablename__ = "predicate_subject_kind"

    predicate_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "predicate_version.id",
            ondelete="RESTRICT",
            name="fk_predicate_subject_kind_version",
        ),
        primary_key=True,
    )
    entity_kind: Mapped[EntityKind] = mapped_column(
        enum_type(EntityKind, name="predicate_subject_entity_kind"),
        primary_key=True,
    )

    predicate_version: Mapped[PredicateVersion] = relationship(
        back_populates="subject_kinds"
    )


class PredicateObjectKind(Immutable, Base):
    """An entity kind allowed in the object position."""

    __tablename__ = "predicate_object_kind"

    predicate_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "predicate_version.id",
            ondelete="RESTRICT",
            name="fk_predicate_object_kind_version",
        ),
        primary_key=True,
    )
    entity_kind: Mapped[EntityKind] = mapped_column(
        enum_type(EntityKind, name="predicate_object_entity_kind"),
        primary_key=True,
    )

    predicate_version: Mapped[PredicateVersion] = relationship(
        back_populates="object_kinds"
    )


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    """The Python source of truth used to seed a PredicateVersion."""

    slug: str
    version: int
    label: str
    description: str
    target_kind: PredicateTarget
    temporal_mode: TemporalMode
    subject_kinds: tuple[EntityKind, ...]
    object_kinds: tuple[EntityKind, ...] = ()
    value_model: type[PredicateValue] | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.slug) is None:
            raise ValueError(f"invalid predicate slug: {self.slug!r}")
        if self.version < 1:
            raise ValueError("predicate versions begin at 1")
        if not self.subject_kinds:
            raise ValueError("a predicate must allow at least one subject kind")
        if self.target_kind is PredicateTarget.ENTITY:
            if not self.object_kinds or self.value_model is not None:
                raise ValueError(
                    "entity predicates require object kinds and no value model"
                )
        elif self.object_kinds or self.value_model is None:
            raise ValueError(
                "value predicates require a value model and no object kinds"
            )

    @property
    def predicate_id(self) -> uuid.UUID:
        return uuid.uuid5(_PREDICATE_ID_NAMESPACE, self.slug)

    @property
    def predicate_version_id(self) -> uuid.UUID:
        return uuid.uuid5(
            _PREDICATE_VERSION_ID_NAMESPACE,
            f"{self.slug}@{self.version}",
        )

    @property
    def value_model_path(self) -> str | None:
        if self.value_model is None:
            return None
        return f"{self.value_model.__module__}.{self.value_model.__qualname__}"

    @property
    def value_schema(self) -> dict[str, Any] | None:
        if self.value_model is None:
            return None
        return self.value_model.model_json_schema()

    @property
    def schema_hash(self) -> str:
        return canonical_json_sha256(
            {
                "slug": self.slug,
                "version": self.version,
                "target_kind": self.target_kind,
                "temporal_mode": self.temporal_mode,
                "subject_kinds": sorted(self.subject_kinds),
                "object_kinds": sorted(self.object_kinds),
                "value_model_path": self.value_model_path,
                "value_schema": self.value_schema,
            }
        )

    def validate_value(
        self,
        value: PredicateValue | Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if self.target_kind is PredicateTarget.ENTITY:
            if value is not None:
                raise ValueError(f"{self.slug} takes an entity object, not a value")
            return None
        if value is None or self.value_model is None:
            raise ValueError(f"{self.slug} requires a value")

        return self.value_model.model_validate(value).model_dump(mode="json")


PREDICATE_SPECS: tuple[PredicateSpec, ...] = (
    PredicateSpec(
        slug="participated_in",
        version=1,
        label="Participated in",
        description="The subject participated in the event represented by the object.",
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.OPTIONAL,
        subject_kinds=(EntityKind.PERSON, EntityKind.ORGANIZATION),
        object_kinds=(EntityKind.EVENT,),
    ),
    PredicateSpec(
        slug="endorsed",
        version=1,
        label="Endorsed",
        description="The subject publicly endorsed the object.",
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.OPTIONAL,
        subject_kinds=(EntityKind.PERSON, EntityKind.ORGANIZATION),
        object_kinds=(
            EntityKind.PERSON,
            EntityKind.ORGANIZATION,
            EntityKind.OPTION,
        ),
    ),
    PredicateSpec(
        slug="candidate_in",
        version=1,
        label="Candidate in",
        description="The subject was or is a candidate in the contest.",
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.OPTIONAL,
        subject_kinds=(EntityKind.PERSON,),
        object_kinds=(EntityKind.CONTEST,),
    ),
    PredicateSpec(
        slug="event_kind",
        version=1,
        label="Event kind",
        description="The normalized kind assigned to a political event.",
        target_kind=PredicateTarget.VALUE,
        temporal_mode=TemporalMode.TIMELESS,
        subject_kinds=(EntityKind.EVENT,),
        value_model=EventKindValue,
    ),
    PredicateSpec(
        slug="public_statement",
        version=1,
        label="Public statement",
        description="A normalized policy or political position stated by the subject.",
        target_kind=PredicateTarget.VALUE,
        temporal_mode=TemporalMode.REQUIRED,
        subject_kinds=(EntityKind.PERSON, EntityKind.ORGANIZATION),
        value_model=PublicStatementValue,
    ),
)

_PREDICATE_SPECS_BY_KEY = {(spec.slug, spec.version): spec for spec in PREDICATE_SPECS}
_PREDICATE_SPECS_BY_VERSION_ID = {
    spec.predicate_version_id: spec for spec in PREDICATE_SPECS
}

if len(_PREDICATE_SPECS_BY_KEY) != len(PREDICATE_SPECS) or len(
    _PREDICATE_SPECS_BY_VERSION_ID
) != len(PREDICATE_SPECS):
    raise RuntimeError("duplicate predicate slug and version in PREDICATE_SPECS")


def get_predicate_spec(slug: str, version: int = 1) -> PredicateSpec:
    try:
        return _PREDICATE_SPECS_BY_KEY[(slug, version)]
    except KeyError as error:
        raise KeyError(f"unknown predicate version: {slug}@{version}") from error


def get_predicate_spec_by_id(predicate_version_id: uuid.UUID) -> PredicateSpec:
    try:
        return _PREDICATE_SPECS_BY_VERSION_ID[predicate_version_id]
    except KeyError as error:
        raise KeyError(
            f"unknown predicate version ID: {predicate_version_id}"
        ) from error


def seed_predicates(session: Session) -> list[PredicateVersion]:
    """Insert missing catalog rows without rewriting historical versions.

    A changed schema with the same slug and version is treated as an error. Bump
    the PredicateSpec version whenever its semantic contract changes.
    """

    seeded_versions: list[PredicateVersion] = []
    for spec in PREDICATE_SPECS:
        predicate = session.get(Predicate, spec.predicate_id)
        if predicate is None:
            predicate = Predicate(
                id=spec.predicate_id,
                slug=spec.slug,
                label=spec.label,
                description=spec.description,
            )
            session.add(predicate)
        elif predicate.slug != spec.slug:
            raise ValueError(
                f"predicate UUID collision for {spec.slug}: {predicate.slug}"
            )
        else:
            predicate.label = spec.label
            predicate.description = spec.description

        version = session.get(PredicateVersion, spec.predicate_version_id)
        if version is None:
            version = PredicateVersion(
                id=spec.predicate_version_id,
                predicate_id=spec.predicate_id,
                version=spec.version,
                target_kind=spec.target_kind,
                temporal_mode=spec.temporal_mode,
                value_model_path=spec.value_model_path,
                value_schema=spec.value_schema,
                schema_hash=spec.schema_hash,
            )
            session.add(version)
            session.add_all(
                PredicateSubjectKind(
                    predicate_version_id=spec.predicate_version_id,
                    entity_kind=entity_kind,
                )
                for entity_kind in spec.subject_kinds
            )
            session.add_all(
                PredicateObjectKind(
                    predicate_version_id=spec.predicate_version_id,
                    entity_kind=entity_kind,
                )
                for entity_kind in spec.object_kinds
            )
        else:
            stored_contract = (
                version.predicate_id,
                version.version,
                version.target_kind,
                version.temporal_mode,
                version.value_model_path,
                version.schema_hash,
            )
            expected_contract = (
                spec.predicate_id,
                spec.version,
                spec.target_kind,
                spec.temporal_mode,
                spec.value_model_path,
                spec.schema_hash,
            )
            if stored_contract != expected_contract:
                raise ValueError(
                    f"predicate contract changed without a version bump: "
                    f"{spec.slug}@{spec.version}"
                )

            stored_subject_kinds = set(
                session.scalars(
                    select(PredicateSubjectKind.entity_kind).where(
                        PredicateSubjectKind.predicate_version_id
                        == spec.predicate_version_id
                    )
                )
            )
            stored_object_kinds = set(
                session.scalars(
                    select(PredicateObjectKind.entity_kind).where(
                        PredicateObjectKind.predicate_version_id
                        == spec.predicate_version_id
                    )
                )
            )
            if stored_subject_kinds != set(spec.subject_kinds):
                raise ValueError(
                    f"stored subject kinds differ for {spec.slug}@{spec.version}"
                )
            if stored_object_kinds != set(spec.object_kinds):
                raise ValueError(
                    f"stored object kinds differ for {spec.slug}@{spec.version}"
                )

        seeded_versions.append(version)

    session.flush()
    return seeded_versions
