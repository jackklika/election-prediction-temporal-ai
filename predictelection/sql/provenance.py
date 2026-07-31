from __future__ import annotations

from enum import StrEnum
from typing import Any
import uuid

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column, relationship

from predictelection.sql.base import (
    Base,
    Immutable,
    created_at_timestamp,
    enum_type,
    nullable_jsonb,
    utc_timestamp,
    uuid_primary_key,
)
from predictelection.sql.entity import Entity


class SourceKind(StrEnum):
    WEB_PAGE = "web_page"
    DOCUMENT = "document"
    VIDEO = "video"
    API_RESPONSE = "api_response"
    DATASET = "dataset"
    SOCIAL_MEDIA = "social_media"
    OTHER = "other"


class ArtifactDerivationKind(StrEnum):
    EXTRACTED_TEXT = "extracted_text"
    OCR = "ocr"
    TRANSCRIPT = "transcript"
    NORMALIZED_DATA = "normalized_data"
    THUMBNAIL = "thumbnail"
    OTHER = "other"


class ResearchRunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Source(Base):
    """A logical publication, URL, endpoint, dataset, or media item."""

    __tablename__ = "source"

    id: Mapped[uuid_primary_key]
    kind: Mapped[SourceKind] = mapped_column(enum_type(SourceKind, name="source_kind"))
    canonical_url: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    publisher_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    publisher_name: Mapped[str | None] = mapped_column(Text)
    author_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT"),
        name="author_entity_id",
    )
    """Who wrote it, when that is a person rather than an outlet.

    For an opinion column the author is the most important attribute, because
    the claims extracted from it are that person's judgements — recorded as
    `assessed` claims whose subject is them. A plain FK rather than reifying
    Source as an Entity: it answers "everything this author wrote" without
    giving every source a second identity to resolve.
    """

    author_name: Mapped[str | None] = mapped_column(Text)
    """The byline as printed, kept even when the author is not yet resolved."""
    created_at: Mapped[created_at_timestamp]

    publisher: Mapped[Entity | None] = relationship(foreign_keys=[publisher_id])
    author: Mapped[Entity | None] = relationship(foreign_keys=[author_id])
    snapshots: Mapped[list[SourceSnapshot]] = relationship(
        back_populates="source",
        passive_deletes="all",
    )

    __table_args__ = (
        UniqueConstraint("canonical_url", name="uq_source_canonical_url"),
        Index("ix_source_publisher_id", "publisher_id"),
        Index("ix_source_author_entity_id", "author_entity_id"),
    )


class Artifact(Immutable, Base):
    """Content-addressed bytes retained in S3 or another durable store.

    storage_uri must be a durable object address such as an s3:// URI, not a
    temporary presigned URL. storage_version_id records the S3 version when
    bucket versioning is enabled.
    """

    __tablename__ = "artifact"

    id: Mapped[uuid_primary_key]
    sha256: Mapped[str] = mapped_column(String(64))
    storage_uri: Mapped[str] = mapped_column(Text)
    storage_version_id: Mapped[str | None] = mapped_column(Text)
    byte_length: Mapped[int] = mapped_column(BigInteger)
    media_type: Mapped[str | None] = mapped_column(String(255))
    original_filename: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[created_at_timestamp]

    __table_args__ = (
        UniqueConstraint("sha256", name="uq_artifact_sha256"),
        UniqueConstraint(
            "storage_uri",
            "storage_version_id",
            name="uq_artifact_storage_object",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint(
            "sha256 ~ '^[0-9a-f]{64}$'",
            name="sha256_lowercase_hex",
        ),
        CheckConstraint("byte_length >= 0", name="byte_length_nonnegative"),
        CheckConstraint("storage_uri <> ''", name="storage_uri_nonempty"),
    )


class SourceSnapshot(Immutable, Base):
    """A source observed at a particular immutable artifact version.

    One row per observation, not per distinct content. Re-fetching a source and
    finding the same bytes is itself evidence — "this page still said X on
    2026-07-30" — so retrieved_at participates in the identity.
    """

    __tablename__ = "source_snapshot"

    id: Mapped[uuid_primary_key]
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source.id", ondelete="RESTRICT")
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("artifact.id", ondelete="RESTRICT")
    )
    retrieved_at: Mapped[utc_timestamp]
    published_at: Mapped[utc_timestamp | None]
    retrieval_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[created_at_timestamp]

    source: Mapped[Source] = relationship(back_populates="snapshots")
    artifact: Mapped[Artifact] = relationship(foreign_keys=[artifact_id])

    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "artifact_id",
            "retrieved_at",
            name="uq_source_snapshot_observation",
        ),
        Index("ix_source_snapshot_artifact_id", "artifact_id"),
        Index("ix_source_snapshot_source_retrieved", "source_id", "retrieved_at"),
    )


class ResearchRun(Base):
    """A research or ingestion execution, whether or not Temporal ran it."""

    __tablename__ = "research_run"

    id: Mapped[uuid_primary_key]
    idempotency_key: Mapped[str] = mapped_column(String(255))
    workflow_id: Mapped[str | None] = mapped_column(String(255))
    workflow_run_id: Mapped[str | None] = mapped_column(String(255))
    task_type: Mapped[str] = mapped_column(String(100))
    subject_entity_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entity.id", ondelete="RESTRICT")
    )
    status: Mapped[ResearchRunStatus] = mapped_column(
        enum_type(ResearchRunStatus, name="research_run_status"),
        default=ResearchRunStatus.RUNNING,
        server_default=text(f"'{ResearchRunStatus.RUNNING.value}'"),
    )
    input_data: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    raw_output: Mapped[dict[str, Any] | None] = mapped_column(
        MutableDict.as_mutable(nullable_jsonb())
    )
    agent_name: Mapped[str | None] = mapped_column(String(255))
    model_id: Mapped[str | None] = mapped_column(String(255))
    prompt_version: Mapped[str | None] = mapped_column(String(255))
    output_schema_version: Mapped[str | None] = mapped_column(String(255))
    started_at: Mapped[utc_timestamp] = mapped_column(server_default=func.now())
    completed_at: Mapped[utc_timestamp | None]
    error_message: Mapped[str | None] = mapped_column(Text)

    subject_entity: Mapped[Entity | None] = relationship(
        foreign_keys=[subject_entity_id]
    )
    inputs: Mapped[list[ResearchRunInput]] = relationship(
        back_populates="run",
        passive_deletes="all",
    )

    __table_args__ = (
        UniqueConstraint(
            "idempotency_key",
            name="uq_research_run_idempotency_key",
        ),
        CheckConstraint(
            """
            (workflow_id IS NULL AND workflow_run_id IS NULL)
            OR
            (workflow_id IS NOT NULL AND workflow_run_id IS NOT NULL)
            """,
            name="temporal_identity_complete",
        ),
        CheckConstraint(
            "completed_at IS NULL OR completed_at >= started_at",
            name="completed_after_started",
        ),
        CheckConstraint(
            """
            (
                status = 'running'
                AND completed_at IS NULL
                AND error_message IS NULL
            )
            OR
            (
                status = 'succeeded'
                AND completed_at IS NOT NULL
                AND error_message IS NULL
            )
            OR
            (
                status = 'failed'
                AND completed_at IS NOT NULL
                AND error_message IS NOT NULL
            )
            OR
            (
                status = 'cancelled'
                AND completed_at IS NOT NULL
            )
            """,
            name="status_matches_outcome",
        ),
        Index(
            "uq_research_run_temporal_execution",
            "workflow_id",
            "workflow_run_id",
            unique=True,
            postgresql_where=text(
                "workflow_id IS NOT NULL AND workflow_run_id IS NOT NULL"
            ),
        ),
        Index("ix_research_run_subject_entity_id", "subject_entity_id"),
        Index("ix_research_run_status_started_at", "status", "started_at"),
    )


class ResearchRunInput(Immutable, Base):
    """An exact source snapshot consumed by a research run."""

    __tablename__ = "research_run_input"

    research_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("research_run.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    source_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_snapshot.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    role: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[created_at_timestamp]

    run: Mapped[ResearchRun] = relationship(back_populates="inputs")
    source_snapshot: Mapped[SourceSnapshot] = relationship(
        foreign_keys=[source_snapshot_id]
    )

    __table_args__ = (
        CheckConstraint("role <> ''", name="role_nonempty"),
        Index("ix_research_run_input_source_snapshot_id", "source_snapshot_id"),
    )


class ArtifactDerivation(Immutable, Base):
    """Lineage from an original artifact to OCR, text, or normalized output."""

    __tablename__ = "artifact_derivation"

    parent_artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("artifact.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    derived_artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("artifact.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    kind: Mapped[ArtifactDerivationKind] = mapped_column(
        enum_type(ArtifactDerivationKind, name="artifact_derivation_kind"),
        primary_key=True,
    )
    research_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("research_run.id", ondelete="RESTRICT")
    )
    processor_name: Mapped[str] = mapped_column(String(255))
    processor_version: Mapped[str | None] = mapped_column(String(255))
    parameters: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[created_at_timestamp]

    parent_artifact: Mapped[Artifact] = relationship(foreign_keys=[parent_artifact_id])
    derived_artifact: Mapped[Artifact] = relationship(
        foreign_keys=[derived_artifact_id]
    )
    research_run: Mapped[ResearchRun | None] = relationship(
        foreign_keys=[research_run_id]
    )

    __table_args__ = (
        CheckConstraint(
            "parent_artifact_id <> derived_artifact_id",
            name="different_artifacts",
        ),
        CheckConstraint("processor_name <> ''", name="processor_name_nonempty"),
        Index("ix_artifact_derivation_derived_id", "derived_artifact_id"),
        Index("ix_artifact_derivation_research_run_id", "research_run_id"),
    )
