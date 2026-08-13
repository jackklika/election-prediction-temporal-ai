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
    CanonicalDecimal,
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
    """What a claim points at besides its subject.

    QUALIFIED is the n-ary case: subject, object, *and* a payload. The formal
    definition of a knowledge graph admits n-ary facts alongside binary triples,
    and the alternative — reifying the relationship as its own entity — roughly
    doubles the graph and produces entities with no canonical name, which cannot
    be resolved on a re-scrape. A moderator and a candidate both "participated
    in" a debate; only the role separates them, and it is not a thing in the
    world that deserves its own identity.
    """

    ENTITY = "entity"
    VALUE = "value"
    QUALIFIED = "qualified"


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


class EventOccurrenceStatus(StrEnum):
    SCHEDULED = "scheduled"
    OCCURRED = "occurred"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class ParticipationRole(StrEnum):
    """Why someone was at an event. Without it, a moderator reads as a debater."""

    CANDIDATE = "candidate"
    MODERATOR = "moderator"
    PANELIST = "panelist"
    HOST = "host"
    OTHER = "other"


class EndorsementStrength(StrEnum):
    FULL = "full"
    QUALIFIED = "qualified"
    """Backed with reservations stated by the endorser."""

    WITHDRAWN = "withdrawn"


class ContestStage(StrEnum):
    PRIMARY = "primary"
    RUNOFF = "runoff"
    GENERAL = "general"
    SPECIAL = "special"
    CAUCUS = "caucus"


class PredicateValue(BaseModel):
    """Base for the JSON value of one literal predicate."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EventKindValue(PredicateValue):
    kind: PoliticalEventKind


class ParticipationValue(PredicateValue):
    role: ParticipationRole


class EndorsementValue(PredicateValue):
    strength: EndorsementStrength = EndorsementStrength.FULL
    context: str | None = Field(default=None, max_length=500)


class ContestStageValue(PredicateValue):
    stage: ContestStage


class ContestResultValue(PredicateValue):
    """One candidate's result in one contest.

    CanonicalDecimal, not Decimal: Pydantic serializes Decimal to a
    scale-preserving string, so "48.7" and "48.70" would be the same result with
    two fingerprints and would not deduplicate.
    """

    votes: int | None = Field(default=None, ge=0)
    share: CanonicalDecimal | None = Field(default=None, ge=0, le=100)
    place: int | None = Field(default=None, ge=1)
    won: bool | None = None
    """Explicit rather than derived from place: multi-winner contests exist.

    Nullable because a results table states counts and not outcomes, and `False`
    would assert something no source said. Defaulting it to False gave two
    honest writers a contradiction: a vote-count claim silently said "did not
    win" while an outcome claim said "won", and neither superseded the other.
    None means the source stated no outcome; a writer that knows one says so.
    """


class AssessmentValue(PredicateValue):
    """A judgement one party made about another — a critic's read, a rating.

    Stored as a claim about the *assessor*, which is what makes it checkable:
    "Crowley was weakest" is unverifiable, "Murphy assessed Crowley as weakest"
    is a fact about a citable column.
    """

    rating: str = Field(min_length=1, max_length=100)
    basis: str | None = Field(default=None, max_length=500)


class EventOccurrenceValue(PredicateValue):
    """Whether an event happened; *when* lives in the claim's validity interval.

    Every predicate is either (subject, object) or (subject, value), so there is
    no way to state a bare timestamped fact about a subject. Carrying the status
    here gives the claim a real payload while valid_from/valid_to carry the
    schedule with their own precision — which also means a postponement is a new
    claim over a new interval rather than an edit.
    """

    status: EventOccurrenceStatus


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
                target_kind IN ('value', 'qualified')
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
    label: str
    description: str
    target_kind: PredicateTarget
    temporal_mode: TemporalMode
    subject_kinds: tuple[EntityKind, ...]
    object_kinds: tuple[EntityKind, ...] = ()
    value_model: type[PredicateValue] | None = None
    version: int = 1
    """Bump when a contract changes meaning.

    Claims fingerprint against predicate_version_id, so a stored claim keeps
    pointing at the contract it was written under. seed_predicates refuses to
    reinterpret existing claims silently: edit a spec without bumping and it
    raises rather than changing what old rows mean.
    """

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
        elif self.target_kind is PredicateTarget.QUALIFIED:
            if not self.object_kinds or self.value_model is None:
                raise ValueError(
                    "qualified predicates require both object kinds and a value model"
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
        label="Participated in",
        description=(
            "The subject took part in the event, in the stated role. Without the "
            "role a moderator is indistinguishable from a debater."
        ),
        target_kind=PredicateTarget.QUALIFIED,
        temporal_mode=TemporalMode.OPTIONAL,
        subject_kinds=(EntityKind.PERSON, EntityKind.ORGANIZATION),
        object_kinds=(EntityKind.EVENT,),
        value_model=ParticipationValue,
    ),
    PredicateSpec(
        slug="endorsed",
        label="Endorsed",
        description=(
            "The subject publicly backed the object, at the stated strength. A "
            "withdrawn endorsement is a new claim over a later interval."
        ),
        target_kind=PredicateTarget.QUALIFIED,
        temporal_mode=TemporalMode.OPTIONAL,
        subject_kinds=(EntityKind.PERSON, EntityKind.ORGANIZATION, EntityKind.PARTY),
        object_kinds=(EntityKind.PERSON, EntityKind.ORGANIZATION, EntityKind.OPTION),
        value_model=EndorsementValue,
    ),
    PredicateSpec(
        slug="candidate_in",
        label="Candidate in",
        description="The subject was or is a candidate in the contest.",
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.OPTIONAL,
        subject_kinds=(EntityKind.PERSON,),
        object_kinds=(EntityKind.CONTEST,),
    ),
    PredicateSpec(
        slug="event_kind",
        label="Event kind",
        description="The normalized kind assigned to a political event.",
        target_kind=PredicateTarget.VALUE,
        temporal_mode=TemporalMode.TIMELESS,
        subject_kinds=(EntityKind.EVENT,),
        value_model=EventKindValue,
    ),
    PredicateSpec(
        slug="public_statement",
        label="Public statement",
        description="A normalized policy or political position stated by the subject.",
        target_kind=PredicateTarget.VALUE,
        temporal_mode=TemporalMode.REQUIRED,
        subject_kinds=(EntityKind.PERSON, EntityKind.ORGANIZATION),
        value_model=PublicStatementValue,
    ),
    # The structural backbone. Contests, offices, elections, and jurisdictions
    # are all just entities, so the relationships between them exist only as
    # claims — without these predicates there is no way to ask which races share
    # a geography, which is what crosstab and correlation queries are built on.
    PredicateSpec(
        slug="contest_of_election",
        label="Contest of election",
        description="The contest is one of the races decided in the given election.",
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.TIMELESS,
        subject_kinds=(EntityKind.CONTEST,),
        object_kinds=(EntityKind.ELECTION,),
    ),
    PredicateSpec(
        slug="contest_for_office",
        label="Contest for office",
        description="The contest determines who holds the given office.",
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.TIMELESS,
        subject_kinds=(EntityKind.CONTEST,),
        object_kinds=(EntityKind.OFFICE,),
    ),
    PredicateSpec(
        slug="office_for_jurisdiction",
        label="Office for jurisdiction",
        description="The office represents or governs the given jurisdiction.",
        # Optional rather than timeless: districts are redrawn, so the pairing
        # can be true only for a period.
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.OPTIONAL,
        subject_kinds=(EntityKind.OFFICE,),
        object_kinds=(EntityKind.JURISDICTION,),
    ),
    PredicateSpec(
        slug="contest_in_jurisdiction",
        label="Contest in jurisdiction",
        description="The contest is decided by voters of the given jurisdiction.",
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.TIMELESS,
        subject_kinds=(EntityKind.CONTEST,),
        object_kinds=(EntityKind.JURISDICTION,),
    ),
    PredicateSpec(
        slug="market_for_contest",
        label="Market for contest",
        description="The prediction market settles on the outcome of the contest.",
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.TIMELESS,
        subject_kinds=(EntityKind.MARKET,),
        object_kinds=(EntityKind.CONTEST,),
    ),
    PredicateSpec(
        slug="event_about_contest",
        label="Event about contest",
        description="The event concerns the given contest, such as its debate.",
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.TIMELESS,
        subject_kinds=(EntityKind.EVENT,),
        object_kinds=(EntityKind.CONTEST,),
    ),
    PredicateSpec(
        slug="event_in_jurisdiction",
        label="Event in jurisdiction",
        description="The event took place in the given jurisdiction.",
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.TIMELESS,
        subject_kinds=(EntityKind.EVENT,),
        object_kinds=(EntityKind.JURISDICTION,),
    ),
    PredicateSpec(
        slug="event_occurrence",
        label="Event occurrence",
        description="Whether an event is scheduled or happened, over its interval.",
        target_kind=PredicateTarget.VALUE,
        temporal_mode=TemporalMode.REQUIRED,
        subject_kinds=(EntityKind.EVENT,),
        value_model=EventOccurrenceValue,
    ),
    # Outcomes. Without these there is nothing to backtest a theory against.
    # Modelled as claims rather than their own table so recounts and
    # certifications supersede like any other correction, and so a result cites
    # the canvass it came from.
    PredicateSpec(
        slug="contest_result",
        label="Contest result",
        description=(
            "How the subject placed in the contest. Supersede rather than edit "
            "as counts move from election night to certified."
        ),
        target_kind=PredicateTarget.QUALIFIED,
        temporal_mode=TemporalMode.OPTIONAL,
        subject_kinds=(EntityKind.PERSON, EntityKind.ORGANIZATION, EntityKind.OPTION),
        object_kinds=(EntityKind.CONTEST,),
        value_model=ContestResultValue,
    ),
    # Contest structure. A primary and a general are separate contests with
    # different candidates, polls, and outcomes; they meet at the office.
    PredicateSpec(
        slug="contest_stage",
        label="Contest stage",
        description="Which stage of the process this contest is.",
        target_kind=PredicateTarget.VALUE,
        temporal_mode=TemporalMode.TIMELESS,
        subject_kinds=(EntityKind.CONTEST,),
        value_model=ContestStageValue,
    ),
    PredicateSpec(
        slug="contest_party",
        label="Contest party",
        description="The party whose nomination this contest decides.",
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.TIMELESS,
        subject_kinds=(EntityKind.CONTEST,),
        object_kinds=(EntityKind.PARTY,),
    ),
    PredicateSpec(
        slug="advances_to",
        label="Advances to",
        description="The winner of this contest goes on to the object contest.",
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.TIMELESS,
        subject_kinds=(EntityKind.CONTEST,),
        object_kinds=(EntityKind.CONTEST,),
    ),
    PredicateSpec(
        slug="party_affiliation",
        label="Party affiliation",
        description=(
            "The subject belonged to the party. Optional temporal because "
            "people switch, and the graph should keep both."
        ),
        target_kind=PredicateTarget.ENTITY,
        temporal_mode=TemporalMode.OPTIONAL,
        subject_kinds=(EntityKind.PERSON, EntityKind.ORGANIZATION),
        object_kinds=(EntityKind.PARTY,),
    ),
    # Opinion. The subject is the assessor, which is what makes an unverifiable
    # judgement into a checkable fact about a citable source.
    PredicateSpec(
        slug="assessed",
        label="Assessed",
        description=(
            "The subject publicly judged the object. Records that the "
            "assessment was made, not that it was correct."
        ),
        target_kind=PredicateTarget.QUALIFIED,
        temporal_mode=TemporalMode.REQUIRED,
        subject_kinds=(EntityKind.PERSON, EntityKind.ORGANIZATION),
        object_kinds=(
            EntityKind.PERSON,
            EntityKind.ORGANIZATION,
            EntityKind.EVENT,
            EntityKind.CONTEST,
        ),
        value_model=AssessmentValue,
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
