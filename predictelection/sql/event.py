from __future__ import annotations

from typing import Any
import uuid

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column, relationship

from predictelection.sql.base import (
    Base,
    TimePrecision,
    created_at_timestamp,
    enum_type,
    utc_timestamp,
)
from predictelection.sql.claim import Claim
from predictelection.sql.entity import Entity
from predictelection.sql.predicate import PoliticalEventKind


class PoliticalEventProjection(Base):
    """Rebuildable current-state fields derived from accepted claims.

    Unlike claims, this table is deliberately mutable. It is a query projection,
    not an independent source of truth.
    """

    __tablename__ = "political_event_projection"

    entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entity.id", ondelete="CASCADE"),
        primary_key=True,
    )
    event_kind: Mapped[PoliticalEventKind] = mapped_column(
        enum_type(PoliticalEventKind, name="political_event_kind")
    )
    contest_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entity.id", ondelete="SET NULL")
    )
    location_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entity.id", ondelete="SET NULL")
    )
    starts_at: Mapped[utc_timestamp | None]
    starts_at_precision: Mapped[TimePrecision | None] = mapped_column(
        enum_type(TimePrecision, name="event_starts_at_precision")
    )
    ends_at: Mapped[utc_timestamp | None]
    ends_at_precision: Mapped[TimePrecision | None] = mapped_column(
        enum_type(TimePrecision, name="event_ends_at_precision")
    )
    details: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    projection_version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        server_default=text("1"),
    )
    refreshed_at: Mapped[utc_timestamp] = mapped_column(
        server_default=func.now(),
        onupdate=func.now(),
    )

    entity: Mapped[Entity] = relationship(foreign_keys=[entity_id])
    contest: Mapped[Entity | None] = relationship(foreign_keys=[contest_id])
    location: Mapped[Entity | None] = relationship(foreign_keys=[location_id])
    claim_links: Mapped[list[PoliticalEventProjectionClaim]] = relationship(
        back_populates="event_projection",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "ends_at IS NULL OR starts_at IS NULL OR ends_at >= starts_at",
            name="ends_after_starts",
        ),
        CheckConstraint(
            "(starts_at IS NULL) = (starts_at_precision IS NULL)",
            name="starts_at_precision_paired",
        ),
        CheckConstraint(
            "(ends_at IS NULL) = (ends_at_precision IS NULL)",
            name="ends_at_precision_paired",
        ),
        CheckConstraint(
            "projection_version > 0",
            name="projection_version_positive",
        ),
        Index(
            "ix_political_event_projection_kind_starts",
            "event_kind",
            "starts_at",
        ),
        Index("ix_political_event_projection_contest_id", "contest_id"),
        Index("ix_political_event_projection_location_id", "location_id"),
    )


class PoliticalEventProjectionClaim(Base):
    """Trace a projected field back to the accepted claim that supplied it."""

    __tablename__ = "political_event_projection_claim"

    event_entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "political_event_projection.entity_id",
            ondelete="CASCADE",
            name="fk_political_event_projection_claim_event",
        ),
        primary_key=True,
    )
    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claim.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    field_name: Mapped[str] = mapped_column(String(100), primary_key=True)
    created_at: Mapped[created_at_timestamp]

    event_projection: Mapped[PoliticalEventProjection] = relationship(
        back_populates="claim_links"
    )
    claim: Mapped[Claim] = relationship(foreign_keys=[claim_id])

    __table_args__ = (
        CheckConstraint("field_name <> ''", name="field_name_nonempty"),
        Index("ix_political_event_projection_claim_claim_id", "claim_id"),
    )
