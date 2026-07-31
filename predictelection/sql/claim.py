"""Claims: source-independent propositions, and the evidence asserting them.

Only time is modelled as a claim qualifier, and deliberately so. Other context —
where a statement was made, which event it happened at, under what conditions —
is reified: mint an Entity for the context and record a second claim pointing at
it, the way Wikidata models an event rather than hanging free-text qualifiers off
a statement. That keeps the claim fingerprint a function of the proposition
alone, so two extractions of the same fact still collapse to one row.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, Mapping, Self
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from predictelection.sql.base import (
    Base,
    CanonicalDecimal,
    Immutable,
    RecordOrigin,
    TimePrecision,
    canonical_json_sha256,
    created_at_timestamp,
    enum_type,
    insert_sequence,
    nullable_jsonb,
    utc_timestamp,
    uuid_primary_key,
)
from predictelection.sql.entity import Entity
from predictelection.sql.predicate import (
    PredicateSpec,
    PredicateTarget,
    PredicateValue,
    PredicateVersion,
    TemporalMode,
    get_predicate_spec_by_id,
)
from predictelection.sql.provenance import ResearchRun, SourceSnapshot


class EvidenceStance(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    MENTIONS = "mentions"


class EvidenceLocatorKind(StrEnum):
    FULL_SOURCE = "full_source"
    PDF = "pdf"
    WEB = "web"
    VIDEO = "video"
    JSON = "json"


class EvidenceLocatorModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FullSourceLocator(EvidenceLocatorModel):
    kind: Literal[EvidenceLocatorKind.FULL_SOURCE] = EvidenceLocatorKind.FULL_SOURCE


class PdfBoundingBox(EvidenceLocatorModel):
    """Normalized PDF coordinates, with the origin at the top left."""

    page: int = Field(ge=1)
    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _coordinates_are_ordered(self) -> Self:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("PDF bounding-box coordinates must be ordered")
        return self


class PdfEvidenceLocator(EvidenceLocatorModel):
    kind: Literal[EvidenceLocatorKind.PDF] = EvidenceLocatorKind.PDF
    page_start: int = Field(ge=1)
    page_end: int | None = Field(default=None, ge=1)
    bounding_boxes: tuple[PdfBoundingBox, ...] = ()

    @model_validator(mode="after")
    def _pages_are_ordered(self) -> Self:
        page_end = self.page_end or self.page_start
        if page_end < self.page_start:
            raise ValueError("PDF page_end must not precede page_start")
        if any(
            box.page < self.page_start or box.page > page_end
            for box in self.bounding_boxes
        ):
            raise ValueError("PDF bounding boxes must fall within the page range")
        return self


class WebEvidenceLocator(EvidenceLocatorModel):
    kind: Literal[EvidenceLocatorKind.WEB] = EvidenceLocatorKind.WEB
    css_selector: str | None = Field(default=None, min_length=1)
    text_fragment: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _has_anchor(self) -> Self:
        if self.css_selector is None and self.text_fragment is None:
            raise ValueError("web evidence needs a selector or text fragment")
        return self


class VideoEvidenceLocator(EvidenceLocatorModel):
    # CanonicalDecimal, not Decimal: Pydantic serializes Decimal to a
    # scale-preserving string, so 12.5 and 12.50 would be the same instant with
    # two different fingerprints and uq_evidence_anchor_fingerprint would not
    # dedupe them.
    kind: Literal[EvidenceLocatorKind.VIDEO] = EvidenceLocatorKind.VIDEO
    start_seconds: CanonicalDecimal = Field(ge=0)
    end_seconds: CanonicalDecimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _times_are_ordered(self) -> Self:
        if self.end_seconds is not None and self.end_seconds < self.start_seconds:
            raise ValueError("video end_seconds must not precede start_seconds")
        return self


class JsonEvidenceLocator(EvidenceLocatorModel):
    kind: Literal[EvidenceLocatorKind.JSON] = EvidenceLocatorKind.JSON
    json_pointer: str = Field(pattern=r"^(|/.*)$")


EvidenceLocator = (
    FullSourceLocator
    | PdfEvidenceLocator
    | WebEvidenceLocator
    | VideoEvidenceLocator
    | JsonEvidenceLocator
)

_EVIDENCE_LOCATOR_MODELS: dict[
    EvidenceLocatorKind,
    type[EvidenceLocatorModel],
] = {
    EvidenceLocatorKind.FULL_SOURCE: FullSourceLocator,
    EvidenceLocatorKind.PDF: PdfEvidenceLocator,
    EvidenceLocatorKind.WEB: WebEvidenceLocator,
    EvidenceLocatorKind.VIDEO: VideoEvidenceLocator,
    EvidenceLocatorKind.JSON: JsonEvidenceLocator,
}


def normalize_evidence_locator(
    locator: EvidenceLocator | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(locator, BaseModel):
        raw_locator = locator.model_dump(mode="python")
    else:
        raw_locator = dict(locator)

    try:
        locator_kind = EvidenceLocatorKind(raw_locator["kind"])
    except (KeyError, ValueError) as error:
        raise ValueError("evidence locator has an unknown or missing kind") from error
    model = _EVIDENCE_LOCATOR_MODELS[locator_kind]
    return model.model_validate(raw_locator).model_dump(mode="json")


def _validate_temporal_shape(
    *,
    valid_at: datetime | None,
    valid_at_precision: TimePrecision | None,
    valid_from: datetime | None,
    valid_from_precision: TimePrecision | None,
    valid_to: datetime | None,
    valid_to_precision: TimePrecision | None,
) -> None:
    pairs = (
        ("valid_at", valid_at, valid_at_precision),
        ("valid_from", valid_from, valid_from_precision),
        ("valid_to", valid_to, valid_to_precision),
    )
    for name, timestamp, precision in pairs:
        if (timestamp is None) != (precision is None):
            raise ValueError(f"{name} and {name}_precision must be provided together")
        if timestamp is not None and timestamp.utcoffset() is None:
            raise ValueError(f"{name} must be timezone-aware")

    if valid_at is not None and (valid_from is not None or valid_to is not None):
        raise ValueError("valid_at cannot be combined with a validity interval")
    if valid_from is not None and valid_to is not None and valid_to <= valid_from:
        raise ValueError("valid_to must be later than valid_from")


def _validate_payload_shape(
    target_kind: PredicateTarget,
    *,
    object_id: uuid.UUID | None,
    value: object | None,
) -> None:
    """Enforce the object/value shape the predicate's target kind demands.

    Mirrors ck_claim_target_matches_payload so the failure arrives with a useful
    message rather than as a constraint violation at flush.
    """

    if target_kind is PredicateTarget.ENTITY:
        if object_id is None or value is not None:
            raise ValueError("entity claims require an object and no value")
    elif target_kind is PredicateTarget.QUALIFIED:
        if object_id is None or value is None:
            raise ValueError("qualified claims require both an object and a value")
    elif object_id is not None or value is None:
        raise ValueError("value claims require a value and no object")


def build_claim_fingerprint(
    *,
    predicate_version_id: uuid.UUID,
    target_kind: PredicateTarget,
    subject_id: uuid.UUID,
    object_id: uuid.UUID | None = None,
    value: Mapping[str, Any] | None = None,
    valid_at: datetime | None = None,
    valid_at_precision: TimePrecision | None = None,
    valid_from: datetime | None = None,
    valid_from_precision: TimePrecision | None = None,
    valid_to: datetime | None = None,
    valid_to_precision: TimePrecision | None = None,
) -> str:
    """Return the stable semantic identity for an immutable claim."""

    _validate_payload_shape(target_kind, object_id=object_id, value=value)

    _validate_temporal_shape(
        valid_at=valid_at,
        valid_at_precision=valid_at_precision,
        valid_from=valid_from,
        valid_from_precision=valid_from_precision,
        valid_to=valid_to,
        valid_to_precision=valid_to_precision,
    )
    return canonical_json_sha256(
        {
            "predicate_version_id": predicate_version_id,
            "target_kind": target_kind,
            "subject_id": subject_id,
            "object_id": object_id,
            "value": value,
            "valid_at": valid_at,
            "valid_at_precision": valid_at_precision,
            "valid_from": valid_from,
            "valid_from_precision": valid_from_precision,
            "valid_to": valid_to,
            "valid_to_precision": valid_to_precision,
        }
    )


class Claim(Immutable, Base):
    """A source-independent immutable proposition.

    Sources and individual extraction attempts attach through ClaimAssertion.
    Review state is append-only in ReviewDecision rather than stored here.
    """

    __tablename__ = "claim"

    id: Mapped[uuid_primary_key]
    predicate_version_id: Mapped[uuid.UUID] = mapped_column()
    target_kind: Mapped[PredicateTarget] = mapped_column(
        enum_type(PredicateTarget, name="claim_target")
    )
    subject_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    object_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    value: Mapped[dict[str, Any] | None] = mapped_column(nullable_jsonb())
    valid_at: Mapped[utc_timestamp | None]
    valid_at_precision: Mapped[TimePrecision | None] = mapped_column(
        enum_type(TimePrecision, name="claim_valid_at_precision")
    )
    valid_from: Mapped[utc_timestamp | None]
    valid_from_precision: Mapped[TimePrecision | None] = mapped_column(
        enum_type(TimePrecision, name="claim_valid_from_precision")
    )
    valid_to: Mapped[utc_timestamp | None]
    valid_to_precision: Mapped[TimePrecision | None] = mapped_column(
        enum_type(TimePrecision, name="claim_valid_to_precision")
    )
    fingerprint: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[created_at_timestamp]

    # viewonly: the relationship can only synchronize predicate_version_id, so
    # assigning it would leave target_kind unset and break the composite foreign
    # key below. Build claims with new_claim, which sets both.
    predicate_version: Mapped[PredicateVersion] = relationship(
        primaryjoin=lambda: Claim.predicate_version_id == PredicateVersion.id,
        foreign_keys=lambda: [Claim.predicate_version_id],
        viewonly=True,
    )
    subject: Mapped[Entity] = relationship(foreign_keys=[subject_id])
    object: Mapped[Entity | None] = relationship(foreign_keys=[object_id])
    assertions: Mapped[list[ClaimAssertion]] = relationship(
        back_populates="claim",
        passive_deletes="all",
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["predicate_version_id", "target_kind"],
            ["predicate_version.id", "predicate_version.target_kind"],
            name="fk_claim_predicate_version_target",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("fingerprint", name="uq_claim_fingerprint"),
        CheckConstraint(
            """
            (
                target_kind = 'entity'
                AND object_id IS NOT NULL
                AND value IS NULL
            )
            OR
            (
                target_kind = 'value'
                AND object_id IS NULL
                AND value IS NOT NULL
            )
            OR
            (
                target_kind = 'qualified'
                AND object_id IS NOT NULL
                AND value IS NOT NULL
            )
            """,
            name="target_matches_payload",
        ),
        CheckConstraint(
            """
            NOT (
                valid_at IS NOT NULL
                AND (valid_from IS NOT NULL OR valid_to IS NOT NULL)
            )
            """,
            name="point_or_interval",
        ),
        CheckConstraint(
            "valid_to IS NULL OR valid_from IS NULL OR valid_to > valid_from",
            name="valid_interval_order",
        ),
        CheckConstraint(
            "(valid_at IS NULL) = (valid_at_precision IS NULL)",
            name="valid_at_precision_paired",
        ),
        CheckConstraint(
            "(valid_from IS NULL) = (valid_from_precision IS NULL)",
            name="valid_from_precision_paired",
        ),
        CheckConstraint(
            "(valid_to IS NULL) = (valid_to_precision IS NULL)",
            name="valid_to_precision_paired",
        ),
        CheckConstraint(
            "fingerprint ~ '^[0-9a-f]{64}$'",
            name="fingerprint_lowercase_hex",
        ),
        Index("ix_claim_predicate_version_id", "predicate_version_id"),
        Index("ix_claim_subject_predicate", "subject_id", "predicate_version_id"),
        Index("ix_claim_object_predicate", "object_id", "predicate_version_id"),
        Index("ix_claim_valid_at", "valid_at"),
        Index("ix_claim_valid_from", "valid_from"),
    )


def new_claim(
    *,
    predicate: PredicateSpec,
    subject_id: uuid.UUID,
    object_id: uuid.UUID | None = None,
    value: PredicateValue | Mapping[str, Any] | None = None,
    valid_at: datetime | None = None,
    valid_at_precision: TimePrecision | None = None,
    valid_from: datetime | None = None,
    valid_from_precision: TimePrecision | None = None,
    valid_to: datetime | None = None,
    valid_to_precision: TimePrecision | None = None,
) -> Claim:
    """Validate a predicate-specific value and build an immutable claim."""

    serialized_value = _validate_claim_contract(
        predicate=predicate,
        target_kind=predicate.target_kind,
        object_id=object_id,
        value=value,
        valid_at=valid_at,
        valid_at_precision=valid_at_precision,
        valid_from=valid_from,
        valid_from_precision=valid_from_precision,
        valid_to=valid_to,
        valid_to_precision=valid_to_precision,
    )
    fingerprint = build_claim_fingerprint(
        predicate_version_id=predicate.predicate_version_id,
        target_kind=predicate.target_kind,
        subject_id=subject_id,
        object_id=object_id,
        value=serialized_value,
        valid_at=valid_at,
        valid_at_precision=valid_at_precision,
        valid_from=valid_from,
        valid_from_precision=valid_from_precision,
        valid_to=valid_to,
        valid_to_precision=valid_to_precision,
    )
    return Claim(
        predicate_version_id=predicate.predicate_version_id,
        target_kind=predicate.target_kind,
        subject_id=subject_id,
        object_id=object_id,
        value=serialized_value,
        valid_at=valid_at,
        valid_at_precision=valid_at_precision,
        valid_from=valid_from,
        valid_from_precision=valid_from_precision,
        valid_to=valid_to,
        valid_to_precision=valid_to_precision,
        fingerprint=fingerprint,
    )


def _validate_claim_contract(
    *,
    predicate: PredicateSpec,
    target_kind: PredicateTarget,
    object_id: uuid.UUID | None,
    value: PredicateValue | Mapping[str, Any] | None,
    valid_at: datetime | None,
    valid_at_precision: TimePrecision | None,
    valid_from: datetime | None,
    valid_from_precision: TimePrecision | None,
    valid_to: datetime | None,
    valid_to_precision: TimePrecision | None,
) -> dict[str, Any] | None:
    if target_kind is not predicate.target_kind:
        raise ValueError(
            f"claim target does not match {predicate.slug}@{predicate.version}"
        )

    serialized_value = predicate.validate_value(value)
    if target_kind is PredicateTarget.VALUE:
        if object_id is not None:
            raise ValueError(f"{predicate.slug} does not take an object entity")
    elif object_id is None:
        raise ValueError(f"{predicate.slug} requires an object entity")

    _validate_temporal_shape(
        valid_at=valid_at,
        valid_at_precision=valid_at_precision,
        valid_from=valid_from,
        valid_from_precision=valid_from_precision,
        valid_to=valid_to,
        valid_to_precision=valid_to_precision,
    )
    has_time = any(item is not None for item in (valid_at, valid_from, valid_to))
    if predicate.temporal_mode is TemporalMode.TIMELESS and has_time:
        raise ValueError(f"{predicate.slug} does not accept temporal qualifiers")
    if predicate.temporal_mode is TemporalMode.REQUIRED and not has_time:
        raise ValueError(f"{predicate.slug} requires a temporal qualifier")

    return serialized_value


@event.listens_for(Claim, "before_insert")
def _derive_claim_fingerprint(
    mapper: object,
    connection: object,
    claim: Claim,
) -> None:
    del mapper, connection
    predicate = get_predicate_spec_by_id(claim.predicate_version_id)
    claim.value = _validate_claim_contract(
        predicate=predicate,
        target_kind=claim.target_kind,
        object_id=claim.object_id,
        value=claim.value,
        valid_at=claim.valid_at,
        valid_at_precision=claim.valid_at_precision,
        valid_from=claim.valid_from,
        valid_from_precision=claim.valid_from_precision,
        valid_to=claim.valid_to,
        valid_to_precision=claim.valid_to_precision,
    )
    claim.fingerprint = build_claim_fingerprint(
        predicate_version_id=claim.predicate_version_id,
        target_kind=claim.target_kind,
        subject_id=claim.subject_id,
        object_id=claim.object_id,
        value=claim.value,
        valid_at=claim.valid_at,
        valid_at_precision=claim.valid_at_precision,
        valid_from=claim.valid_from,
        valid_from_precision=claim.valid_from_precision,
        valid_to=claim.valid_to,
        valid_to_precision=claim.valid_to_precision,
    )


class EvidenceAnchor(Immutable, Base):
    """A reproducible location within an archived source snapshot."""

    __tablename__ = "evidence_anchor"

    id: Mapped[uuid_primary_key]
    source_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_snapshot.id", ondelete="RESTRICT")
    )
    locator_kind: Mapped[EvidenceLocatorKind] = mapped_column(
        enum_type(EvidenceLocatorKind, name="evidence_locator_kind")
    )
    locator: Mapped[dict[str, Any]] = mapped_column(JSONB)
    excerpt: Mapped[str | None] = mapped_column(Text)
    fingerprint: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[created_at_timestamp]

    source_snapshot: Mapped[SourceSnapshot] = relationship(
        foreign_keys=[source_snapshot_id]
    )

    __table_args__ = (
        UniqueConstraint("fingerprint", name="uq_evidence_anchor_fingerprint"),
        CheckConstraint(
            "fingerprint ~ '^[0-9a-f]{64}$'",
            name="fingerprint_lowercase_hex",
        ),
        CheckConstraint(
            "locator ->> 'kind' = locator_kind",
            name="locator_kind_matches_payload",
        ),
        Index("ix_evidence_anchor_source_snapshot_id", "source_snapshot_id"),
    )


def build_evidence_anchor_fingerprint(
    *,
    source_snapshot_id: uuid.UUID,
    locator: EvidenceLocator | Mapping[str, Any],
    excerpt: str | None,
) -> str:
    normalized_locator = normalize_evidence_locator(locator)
    return canonical_json_sha256(
        {
            "source_snapshot_id": source_snapshot_id,
            "locator": normalized_locator,
            "excerpt": excerpt,
        }
    )


def new_evidence_anchor(
    *,
    source_snapshot_id: uuid.UUID,
    locator: EvidenceLocator | Mapping[str, Any],
    excerpt: str | None = None,
) -> EvidenceAnchor:
    normalized_locator = normalize_evidence_locator(locator)
    return EvidenceAnchor(
        source_snapshot_id=source_snapshot_id,
        locator_kind=EvidenceLocatorKind(normalized_locator["kind"]),
        locator=normalized_locator,
        excerpt=excerpt,
        fingerprint=build_evidence_anchor_fingerprint(
            source_snapshot_id=source_snapshot_id,
            locator=normalized_locator,
            excerpt=excerpt,
        ),
    )


@event.listens_for(EvidenceAnchor, "before_insert")
def _derive_evidence_anchor_fingerprint(
    mapper: object,
    connection: object,
    anchor: EvidenceAnchor,
) -> None:
    del mapper, connection
    anchor.locator = normalize_evidence_locator(anchor.locator)
    anchor.locator_kind = EvidenceLocatorKind(anchor.locator["kind"])
    anchor.fingerprint = build_evidence_anchor_fingerprint(
        source_snapshot_id=anchor.source_snapshot_id,
        locator=anchor.locator,
        excerpt=anchor.excerpt,
    )


class ClaimAssertion(Immutable, Base):
    """One model, human, or import asserting a claim from anchored evidence.

    ontology_aligned records whether the claim's subject and object kinds matched
    the predicate's declared domain at the time of extraction. It lives here
    rather than on Claim for two reasons: Claim is immutable and fingerprinted
    while Entity.kind is not, so a flag stored there would go stale; and this is
    the per-extraction unit an alignment score is computed over.
    """

    __tablename__ = "claim_assertion"

    id: Mapped[uuid_primary_key]
    seq: Mapped[insert_sequence]
    idempotency_key: Mapped[str] = mapped_column(String(255))
    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claim.id", ondelete="RESTRICT")
    )
    evidence_anchor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evidence_anchor.id", ondelete="RESTRICT")
    )
    research_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("research_run.id", ondelete="RESTRICT")
    )
    supersedes_assertion_id: Mapped[uuid.UUID | None] = mapped_column()
    stance: Mapped[EvidenceStance] = mapped_column(
        enum_type(EvidenceStance, name="evidence_stance")
    )
    origin: Mapped[RecordOrigin] = mapped_column(
        enum_type(RecordOrigin, name="claim_assertion_origin")
    )
    asserted_by: Mapped[str | None] = mapped_column(String(255))
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    ontology_aligned: Mapped[bool] = mapped_column(
        default=True,
        server_default=text("true"),
    )
    ontology_violation: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[created_at_timestamp]

    claim: Mapped[Claim] = relationship(back_populates="assertions")
    evidence_anchor: Mapped[EvidenceAnchor] = relationship(
        foreign_keys=[evidence_anchor_id]
    )
    research_run: Mapped[ResearchRun | None] = relationship(
        foreign_keys=[research_run_id]
    )
    supersedes_assertion: Mapped[ClaimAssertion | None] = relationship(
        primaryjoin=lambda: ClaimAssertion.supersedes_assertion_id == ClaimAssertion.id,
        remote_side=lambda: [ClaimAssertion.id],
        foreign_keys=lambda: [ClaimAssertion.supersedes_assertion_id],
        viewonly=True,
    )

    __table_args__ = (
        # Composite, so an assertion can only supersede another assertion about
        # the same claim. MATCH SIMPLE makes this a no-op when the column is NULL.
        ForeignKeyConstraint(
            ["supersedes_assertion_id", "claim_id"],
            ["claim_assertion.id", "claim_assertion.claim_id"],
            name="fk_claim_assertion_supersedes_same_claim",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "idempotency_key",
            name="uq_claim_assertion_idempotency_key",
        ),
        UniqueConstraint("seq", name="uq_claim_assertion_seq"),
        UniqueConstraint(
            "id",
            "claim_id",
            name="uq_claim_assertion_id_claim",
        ),
        UniqueConstraint(
            "supersedes_assertion_id",
            name="uq_claim_assertion_superseded_once",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
        CheckConstraint(
            "supersedes_assertion_id IS NULL OR supersedes_assertion_id <> id",
            name="does_not_supersede_self",
        ),
        CheckConstraint(
            "ontology_aligned = (ontology_violation IS NULL)",
            name="ontology_violation_paired",
        ),
        Index("ix_claim_assertion_claim_id", "claim_id"),
        Index("ix_claim_assertion_claim_seq", "claim_id", "seq"),
        Index("ix_claim_assertion_evidence_anchor_id", "evidence_anchor_id"),
        Index("ix_claim_assertion_research_run_id", "research_run_id"),
        Index(
            "ix_claim_assertion_misaligned",
            "research_run_id",
            postgresql_where=text("ontology_aligned IS FALSE"),
        ),
    )


def check_claim_ontology(
    session: Session,
    *,
    predicate: PredicateSpec,
    subject_id: uuid.UUID,
    object_id: uuid.UUID | None,
) -> str | None:
    """Compare a claim's entity kinds against the predicate's declared domain.

    This is Wikontic's ontology stage, kept advisory. It returns a description of
    the mismatch rather than raising, because hard-excluding misaligned triplets
    costs recall on real extraction output; the caller records the violation on
    the assertion and lets review decide.
    """

    wanted = {subject_id} if object_id is None else {subject_id, object_id}
    kinds = {
        entity_id: kind
        for entity_id, kind in session.execute(
            select(Entity.id, Entity.kind).where(Entity.id.in_(wanted))
        )
    }

    violations: list[str] = []
    subject_kind = kinds.get(subject_id)
    if subject_kind is None:
        violations.append(f"subject entity {subject_id} does not exist")
    elif subject_kind not in predicate.subject_kinds:
        allowed = ", ".join(sorted(predicate.subject_kinds))
        violations.append(f"subject kind {subject_kind} is not one of [{allowed}]")

    if object_id is not None:
        object_kind = kinds.get(object_id)
        if object_kind is None:
            violations.append(f"object entity {object_id} does not exist")
        elif object_kind not in predicate.object_kinds:
            allowed = ", ".join(sorted(predicate.object_kinds))
            violations.append(f"object kind {object_kind} is not one of [{allowed}]")

    if not violations:
        return None
    return f"{predicate.slug}@{predicate.version}: " + "; ".join(violations)


def new_claim_assertion(
    session: Session,
    *,
    claim: Claim,
    evidence_anchor: EvidenceAnchor,
    idempotency_key: str,
    stance: EvidenceStance,
    origin: RecordOrigin,
    research_run_id: uuid.UUID | None = None,
    supersedes_assertion_id: uuid.UUID | None = None,
    asserted_by: str | None = None,
    confidence: Decimal | None = None,
    details: Mapping[str, Any] | None = None,
) -> ClaimAssertion:
    """Assert a claim, flagging any ontology mismatch and queueing it for review.

    Adds the assertion (and, when misaligned, a ReviewTask) to the session, but
    does not flush. The claim need not be persisted yet; both are wired by
    relationship so SQLAlchemy resolves the keys at flush time.
    """

    predicate = get_predicate_spec_by_id(claim.predicate_version_id)
    violation = check_claim_ontology(
        session,
        predicate=predicate,
        subject_id=claim.subject_id,
        object_id=claim.object_id,
    )
    assertion = ClaimAssertion(
        claim=claim,
        evidence_anchor=evidence_anchor,
        idempotency_key=idempotency_key,
        stance=stance,
        origin=origin,
        research_run_id=research_run_id,
        supersedes_assertion_id=supersedes_assertion_id,
        asserted_by=asserted_by,
        confidence=confidence,
        ontology_aligned=violation is None,
        ontology_violation=violation,
        details=dict(details) if details is not None else {},
    )
    session.add(assertion)

    if violation is not None:
        # Deferred: review.py imports this module, so the dependency only works
        # in this direction at call time.
        from predictelection.sql.review import ReviewTask

        session.add(ReviewTask(claim_assertion=assertion, reason=violation))
    return assertion


def ontology_alignment_score(
    session: Session,
    *,
    research_run_id: uuid.UUID | None = None,
) -> float | None:
    """Share of assertions whose entity kinds matched the predicate's domain.

    Wikontic's ontology-entailment metric. Scoped to a research run it becomes a
    quality signal for one extraction pass. None when there is nothing to score.
    """

    statement = select(
        func.count(ClaimAssertion.id),
        func.count(ClaimAssertion.id).filter(ClaimAssertion.ontology_aligned),
    )
    if research_run_id is not None:
        statement = statement.where(ClaimAssertion.research_run_id == research_run_id)
    total, aligned = session.execute(statement).one()
    if not total:
        return None
    return aligned / total


class ClaimSupersession(Immutable, Base):
    """An accepted correction that replaces one immutable claim with another."""

    __tablename__ = "claim_supersession"

    id: Mapped[uuid_primary_key]
    idempotency_key: Mapped[str] = mapped_column(String(255))
    predecessor_claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claim.id", ondelete="RESTRICT")
    )
    successor_claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claim.id", ondelete="RESTRICT")
    )
    origin: Mapped[RecordOrigin] = mapped_column(
        enum_type(RecordOrigin, name="claim_supersession_origin")
    )
    created_by: Mapped[str] = mapped_column(String(255))
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[created_at_timestamp]

    predecessor_claim: Mapped[Claim] = relationship(foreign_keys=[predecessor_claim_id])
    successor_claim: Mapped[Claim] = relationship(foreign_keys=[successor_claim_id])

    __table_args__ = (
        UniqueConstraint(
            "idempotency_key",
            name="uq_claim_supersession_idempotency_key",
        ),
        UniqueConstraint(
            "predecessor_claim_id",
            name="uq_claim_supersession_predecessor",
        ),
        CheckConstraint(
            "predecessor_claim_id <> successor_claim_id",
            name="different_claims",
        ),
        CheckConstraint("created_by <> '' AND reason <> ''", name="audit_nonempty"),
        Index("ix_claim_supersession_successor_claim_id", "successor_claim_id"),
    )
