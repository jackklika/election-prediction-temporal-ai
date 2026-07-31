from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Mapping
import uuid

from pydantic import BaseModel
from sqlalchemy import (
    CheckConstraint,
    Date,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from predictelection.sql.base import (
    Base,
    Immutable,
    RecordOrigin,
    canonical_json_sha256,
    created_at_timestamp,
    enum_type,
    utc_timestamp,
    uuid_primary_key,
)
from predictelection.sql.entity import Entity
from predictelection.sql.provenance import ResearchRun, SourceSnapshot


def build_poll_payload_hash(payload: BaseModel | Mapping[str, Any]) -> str:
    """Hash the complete normalized extraction used to construct a revision."""

    if isinstance(payload, BaseModel):
        value = payload.model_dump(mode="json")
    else:
        value = payload
    return canonical_json_sha256(value)


class Poll(Base):
    """Stable identity for one poll or survey release."""

    __tablename__ = "poll"

    id: Mapped[uuid_primary_key]
    external_namespace: Mapped[str | None] = mapped_column(String(100))
    external_id: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[created_at_timestamp]

    revisions: Mapped[list[PollRevision]] = relationship(back_populates="poll")

    __table_args__ = (
        CheckConstraint(
            "(external_namespace IS NULL) = (external_id IS NULL)",
            name="external_identity_complete",
        ),
        Index(
            "uq_poll_external_identity",
            "external_namespace",
            "external_id",
            unique=True,
            postgresql_where=text(
                "external_namespace IS NOT NULL AND external_id IS NOT NULL"
            ),
        ),
    )


class PollRevision(Immutable, Base):
    """One complete, immutable interpretation of a poll source.

    Human corrections create another row and set supersedes_revision_id.
    Questions, samples, options, and estimates are copied into that new
    revision so the original model output remains reproducible.
    """

    __tablename__ = "poll_revision"

    id: Mapped[uuid_primary_key]
    poll_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("poll.id", ondelete="RESTRICT")
    )
    revision_number: Mapped[int] = mapped_column(Integer)
    supersedes_revision_id: Mapped[uuid.UUID | None] = mapped_column()
    source_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_snapshot.id", ondelete="RESTRICT")
    )
    research_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("research_run.id", ondelete="RESTRICT")
    )
    origin: Mapped[RecordOrigin] = mapped_column(
        enum_type(RecordOrigin, name="poll_revision_origin")
    )
    created_by: Mapped[str | None] = mapped_column(String(255))
    payload_hash: Mapped[str] = mapped_column(String(64))
    pollster_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    sponsor_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    fieldwork_started_on: Mapped[date | None] = mapped_column(Date)
    fieldwork_ended_on: Mapped[date | None] = mapped_column(Date)
    published_at: Mapped[utc_timestamp | None]
    collection_mode: Mapped[str | None] = mapped_column(String(100))
    methodology: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[created_at_timestamp]

    poll: Mapped[Poll] = relationship(back_populates="revisions")
    supersedes_revision: Mapped[PollRevision | None] = relationship(
        primaryjoin=lambda: PollRevision.supersedes_revision_id == PollRevision.id,
        remote_side=lambda: [PollRevision.id],
        foreign_keys=[supersedes_revision_id],
        viewonly=True,
    )
    source_snapshot: Mapped[SourceSnapshot] = relationship(
        foreign_keys=[source_snapshot_id]
    )
    research_run: Mapped[ResearchRun | None] = relationship(
        foreign_keys=[research_run_id]
    )
    pollster: Mapped[Entity | None] = relationship(foreign_keys=[pollster_id])
    sponsor: Mapped[Entity | None] = relationship(foreign_keys=[sponsor_id])
    samples: Mapped[list[PollSample]] = relationship(
        back_populates="poll_revision",
        order_by="PollSample.position",
        lazy="selectin",
        passive_deletes="all",
    )
    questions: Mapped[list[PollQuestion]] = relationship(
        back_populates="poll_revision",
        order_by="PollQuestion.position",
        lazy="selectin",
        passive_deletes="all",
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["supersedes_revision_id", "poll_id"],
            ["poll_revision.id", "poll_revision.poll_id"],
            name="fk_poll_revision_supersedes_same_poll",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "id",
            "poll_id",
            name="uq_poll_revision_id_poll",
        ),
        UniqueConstraint(
            "poll_id",
            "revision_number",
            name="uq_poll_revision_number",
        ),
        UniqueConstraint(
            "poll_id",
            "payload_hash",
            name="uq_poll_revision_payload",
        ),
        UniqueConstraint(
            "supersedes_revision_id",
            name="uq_poll_revision_superseded_once",
        ),
        CheckConstraint("revision_number > 0", name="revision_number_positive"),
        CheckConstraint(
            "supersedes_revision_id IS NULL OR supersedes_revision_id <> id",
            name="does_not_supersede_self",
        ),
        CheckConstraint(
            """
            fieldwork_ended_on IS NULL
            OR fieldwork_started_on IS NULL
            OR fieldwork_ended_on >= fieldwork_started_on
            """,
            name="fieldwork_order",
        ),
        CheckConstraint(
            "payload_hash ~ '^[0-9a-f]{64}$'",
            name="payload_hash_lowercase_hex",
        ),
        Index("ix_poll_revision_pollster_id", "pollster_id"),
        Index("ix_poll_revision_sponsor_id", "sponsor_id"),
        Index("ix_poll_revision_source_snapshot_id", "source_snapshot_id"),
        Index("ix_poll_revision_research_run_id", "research_run_id"),
    )


def new_poll_revision(
    *,
    payload: BaseModel | Mapping[str, Any],
    **fields: Any,
) -> PollRevision:
    """Build a revision whose payload_hash is derived, not asserted.

    uq_poll_revision_payload is the only thing stopping the same extraction being
    stored twice, and its CHECK accepts any 64-hex string, so a caller-supplied
    hash can silently defeat it. Constructing PollRevision directly is
    unsupported for that reason.
    """

    if "payload_hash" in fields:
        raise ValueError("payload_hash is derived from payload")
    return PollRevision(payload_hash=build_poll_payload_hash(payload), **fields)


class PollSample(Immutable, Base):
    """The overall population or one crosstab represented in a revision."""

    __tablename__ = "poll_sample"

    id: Mapped[uuid_primary_key]
    poll_revision_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("poll_revision.id", ondelete="RESTRICT")
    )
    position: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(Text)
    population: Mapped[str] = mapped_column(String(100))
    sample_size: Mapped[int | None] = mapped_column(Integer)
    margin_of_error: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    attributes: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[created_at_timestamp]

    poll_revision: Mapped[PollRevision] = relationship(back_populates="samples")

    __table_args__ = (
        UniqueConstraint(
            "poll_revision_id",
            "position",
            name="uq_poll_sample_position",
        ),
        UniqueConstraint(
            "id",
            "poll_revision_id",
            name="uq_poll_sample_id_revision",
        ),
        CheckConstraint("position >= 0", name="position_nonnegative"),
        CheckConstraint("label <> '' AND population <> ''", name="labels_nonempty"),
        CheckConstraint(
            "sample_size IS NULL OR sample_size > 0",
            name="sample_size_positive",
        ),
        CheckConstraint(
            "margin_of_error IS NULL OR margin_of_error >= 0",
            name="margin_of_error_nonnegative",
        ),
    )


class PollQuestion(Immutable, Base):
    """A question exactly as represented in a poll revision."""

    __tablename__ = "poll_question"

    id: Mapped[uuid_primary_key]
    poll_revision_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("poll_revision.id", ondelete="RESTRICT")
    )
    contest_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    position: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[created_at_timestamp]

    poll_revision: Mapped[PollRevision] = relationship(back_populates="questions")
    contest: Mapped[Entity | None] = relationship(foreign_keys=[contest_id])
    options: Mapped[list[PollOption]] = relationship(
        back_populates="question",
        order_by="PollOption.position",
        lazy="selectin",
        passive_deletes="all",
    )

    __table_args__ = (
        UniqueConstraint(
            "poll_revision_id",
            "position",
            name="uq_poll_question_position",
        ),
        UniqueConstraint(
            "id",
            "poll_revision_id",
            name="uq_poll_question_id_revision",
        ),
        CheckConstraint("position >= 0", name="position_nonnegative"),
        CheckConstraint("text <> ''", name="text_nonempty"),
        Index("ix_poll_question_contest_id", "contest_id"),
    )


class PollOption(Immutable, Base):
    """A response option, separated from estimates across samples.

    poll_revision_id is denormalized from the question so that PollEstimate can
    prove, in the database, that its option and sample come from the same
    revision. Build these with new_poll_option rather than by hand.
    """

    __tablename__ = "poll_option"

    id: Mapped[uuid_primary_key]
    question_id: Mapped[uuid.UUID] = mapped_column()
    poll_revision_id: Mapped[uuid.UUID] = mapped_column()
    position: Mapped[int] = mapped_column(Integer)
    choice_entity_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    label: Mapped[str] = mapped_column(Text)
    created_at: Mapped[created_at_timestamp]

    question: Mapped[PollQuestion] = relationship(back_populates="options")
    choice_entity: Mapped[Entity | None] = relationship(foreign_keys=[choice_entity_id])
    estimates: Mapped[list[PollEstimate]] = relationship(
        back_populates="option",
        passive_deletes="all",
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["question_id", "poll_revision_id"],
            ["poll_question.id", "poll_question.poll_revision_id"],
            name="fk_poll_option_question_same_revision",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "question_id",
            "position",
            name="uq_poll_option_position",
        ),
        UniqueConstraint(
            "id",
            "poll_revision_id",
            name="uq_poll_option_id_revision",
        ),
        CheckConstraint("position >= 0", name="position_nonnegative"),
        CheckConstraint("label <> ''", name="label_nonempty"),
        Index("ix_poll_option_choice_entity_id", "choice_entity_id"),
    )


def new_poll_option(
    *,
    question: PollQuestion,
    position: int,
    label: str,
    choice_entity_id: uuid.UUID | None = None,
) -> PollOption:
    """Build an option that carries its question's revision."""

    return PollOption(
        question_id=question.id,
        poll_revision_id=question.poll_revision_id,
        position=position,
        label=label,
        choice_entity_id=choice_entity_id,
    )


class PollEstimate(Immutable, Base):
    """A measured option result for one question and one sample/crosstab.

    The two composite foreign keys below are the whole point of this table's
    shape: option and sample each reach poll_revision by an independent path, so
    without a shared poll_revision_id an estimate could pair an option from one
    revision with a crosstab from another. Build these with new_poll_estimate.
    """

    __tablename__ = "poll_estimate"

    id: Mapped[uuid_primary_key]
    option_id: Mapped[uuid.UUID] = mapped_column()
    sample_id: Mapped[uuid.UUID] = mapped_column()
    poll_revision_id: Mapped[uuid.UUID] = mapped_column()
    percentage: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    response_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[created_at_timestamp]

    option: Mapped[PollOption] = relationship(back_populates="estimates")
    sample: Mapped[PollSample] = relationship(
        primaryjoin=lambda: PollEstimate.sample_id == PollSample.id,
        foreign_keys=lambda: [PollEstimate.sample_id],
        viewonly=True,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["option_id", "poll_revision_id"],
            ["poll_option.id", "poll_option.poll_revision_id"],
            name="fk_poll_estimate_option_same_revision",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["sample_id", "poll_revision_id"],
            ["poll_sample.id", "poll_sample.poll_revision_id"],
            name="fk_poll_estimate_sample_same_revision",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "option_id",
            "sample_id",
            name="uq_poll_estimate_option_sample",
        ),
        CheckConstraint(
            "percentage IS NOT NULL OR response_count IS NOT NULL",
            name="measurement_present",
        ),
        CheckConstraint(
            "percentage IS NULL OR (percentage >= 0 AND percentage <= 100)",
            name="percentage_range",
        ),
        CheckConstraint(
            "response_count IS NULL OR response_count >= 0",
            name="response_count_nonnegative",
        ),
        Index("ix_poll_estimate_sample_id", "sample_id"),
    )


def new_poll_estimate(
    *,
    option: PollOption,
    sample: PollSample,
    percentage: Decimal | None = None,
    response_count: int | None = None,
) -> PollEstimate:
    """Build an estimate, refusing an option and sample from different revisions."""

    if option.poll_revision_id != sample.poll_revision_id:
        raise ValueError(
            "poll estimates cannot join an option and sample from different revisions"
        )
    return PollEstimate(
        option_id=option.id,
        sample_id=sample.id,
        poll_revision_id=option.poll_revision_id,
        percentage=percentage,
        response_count=response_count,
    )


class PollAverage(Base):
    """Stable identity for an aggregator's series for one contest."""

    __tablename__ = "poll_average"

    id: Mapped[uuid_primary_key]
    aggregator_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    contest_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    series_name: Mapped[str | None] = mapped_column(String(255))
    external_namespace: Mapped[str | None] = mapped_column(String(100))
    external_id: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[created_at_timestamp]

    aggregator: Mapped[Entity] = relationship(foreign_keys=[aggregator_id])
    contest: Mapped[Entity] = relationship(foreign_keys=[contest_id])
    revisions: Mapped[list[PollAverageRevision]] = relationship(
        back_populates="poll_average"
    )

    __table_args__ = (
        UniqueConstraint(
            "aggregator_id",
            "contest_id",
            "series_name",
            name="uq_poll_average_series",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint(
            "(external_namespace IS NULL) = (external_id IS NULL)",
            name="external_identity_complete",
        ),
        Index(
            "uq_poll_average_external_identity",
            "external_namespace",
            "external_id",
            unique=True,
            postgresql_where=text(
                "external_namespace IS NOT NULL AND external_id IS NOT NULL"
            ),
        ),
        Index("ix_poll_average_contest_id", "contest_id"),
    )


class PollAverageRevision(Immutable, Base):
    """An immutable snapshot of every estimate in a poll-average series."""

    __tablename__ = "poll_average_revision"

    id: Mapped[uuid_primary_key]
    poll_average_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("poll_average.id", ondelete="RESTRICT")
    )
    revision_number: Mapped[int] = mapped_column(Integer)
    supersedes_revision_id: Mapped[uuid.UUID | None] = mapped_column()
    source_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_snapshot.id", ondelete="RESTRICT")
    )
    research_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("research_run.id", ondelete="RESTRICT")
    )
    as_of: Mapped[utc_timestamp]
    model_version: Mapped[str | None] = mapped_column(String(255))
    payload_hash: Mapped[str] = mapped_column(String(64))
    origin: Mapped[RecordOrigin] = mapped_column(
        enum_type(RecordOrigin, name="poll_average_revision_origin")
    )
    created_by: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[created_at_timestamp]

    poll_average: Mapped[PollAverage] = relationship(back_populates="revisions")
    supersedes_revision: Mapped[PollAverageRevision | None] = relationship(
        primaryjoin=lambda: (
            PollAverageRevision.supersedes_revision_id == PollAverageRevision.id
        ),
        remote_side=lambda: [PollAverageRevision.id],
        foreign_keys=[supersedes_revision_id],
        viewonly=True,
    )
    source_snapshot: Mapped[SourceSnapshot] = relationship(
        foreign_keys=[source_snapshot_id]
    )
    research_run: Mapped[ResearchRun | None] = relationship(
        foreign_keys=[research_run_id]
    )
    estimates: Mapped[list[PollAverageEstimate]] = relationship(
        back_populates="poll_average_revision",
        lazy="selectin",
        passive_deletes="all",
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["supersedes_revision_id", "poll_average_id"],
            ["poll_average_revision.id", "poll_average_revision.poll_average_id"],
            name="fk_poll_average_revision_supersedes_same_series",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "id",
            "poll_average_id",
            name="uq_poll_average_revision_id_series",
        ),
        UniqueConstraint(
            "poll_average_id",
            "revision_number",
            name="uq_poll_average_revision_number",
        ),
        UniqueConstraint(
            "poll_average_id",
            "payload_hash",
            name="uq_poll_average_revision_payload",
        ),
        UniqueConstraint(
            "supersedes_revision_id",
            name="uq_poll_average_revision_superseded_once",
        ),
        CheckConstraint("revision_number > 0", name="revision_number_positive"),
        CheckConstraint(
            "supersedes_revision_id IS NULL OR supersedes_revision_id <> id",
            name="does_not_supersede_self",
        ),
        CheckConstraint(
            "payload_hash ~ '^[0-9a-f]{64}$'",
            name="payload_hash_lowercase_hex",
        ),
        Index("ix_poll_average_revision_source_snapshot_id", "source_snapshot_id"),
        Index("ix_poll_average_revision_research_run_id", "research_run_id"),
        Index("ix_poll_average_revision_as_of", "as_of"),
    )


def new_poll_average_revision(
    *,
    payload: BaseModel | Mapping[str, Any],
    **fields: Any,
) -> PollAverageRevision:
    """Build a series revision whose payload_hash is derived, not asserted."""

    if "payload_hash" in fields:
        raise ValueError("payload_hash is derived from payload")
    return PollAverageRevision(
        payload_hash=build_poll_payload_hash(payload),
        **fields,
    )


class PollAverageEstimate(Immutable, Base):
    """One choice's estimate and optional uncertainty interval."""

    __tablename__ = "poll_average_estimate"

    id: Mapped[uuid_primary_key]
    poll_average_revision_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "poll_average_revision.id",
            ondelete="RESTRICT",
            name="fk_poll_average_estimate_revision",
        )
    )
    choice_entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    percentage: Mapped[Decimal] = mapped_column(Numeric(6, 3))
    lower_bound: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    upper_bound: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    created_at: Mapped[created_at_timestamp]

    poll_average_revision: Mapped[PollAverageRevision] = relationship(
        back_populates="estimates"
    )
    choice_entity: Mapped[Entity] = relationship(foreign_keys=[choice_entity_id])

    __table_args__ = (
        UniqueConstraint(
            "poll_average_revision_id",
            "choice_entity_id",
            name="uq_poll_average_estimate_choice",
        ),
        CheckConstraint(
            "percentage >= 0 AND percentage <= 100",
            name="percentage_range",
        ),
        CheckConstraint(
            "lower_bound IS NULL OR (lower_bound >= 0 AND lower_bound <= 100)",
            name="lower_bound_range",
        ),
        CheckConstraint(
            "upper_bound IS NULL OR (upper_bound >= 0 AND upper_bound <= 100)",
            name="upper_bound_range",
        ),
        CheckConstraint(
            """
            (lower_bound IS NULL OR lower_bound <= percentage)
            AND (upper_bound IS NULL OR percentage <= upper_bound)
            """,
            name="bounds_order",
        ),
        Index("ix_poll_average_estimate_choice_entity_id", "choice_entity_id"),
    )
