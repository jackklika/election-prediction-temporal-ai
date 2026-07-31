from __future__ import annotations

from enum import StrEnum
import uuid

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from predictelection.sql.base import (
    Base,
    Immutable,
    created_at_timestamp,
    enum_type,
    insert_sequence,
    utc_timestamp,
    uuid_primary_key,
)
from predictelection.sql.claim import ClaimAssertion
from predictelection.sql.polling import PollAverageRevision, PollRevision
from predictelection.sql.provenance import ResearchRun


class ReviewOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class ReviewerKind(StrEnum):
    HUMAN = "human"
    AUTOMATED_RULE = "automated_rule"


class ReviewTaskStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


DEFAULT_REVIEW_PRIORITY = 50
"""Midpoint of ck_review_task_priority_range, so either direction is available."""


class ReviewDecision(Immutable, Base):
    """An append-only decision about one claim or domain-data revision.

    Current state is the decision with the highest seq for the target. Pending is
    represented by the absence of a decision, so prior decisions are never
    overwritten. Order by seq rather than created_at: now() is evaluated once per
    transaction, so decisions written together are indistinguishable by time.
    """

    __tablename__ = "review_decision"

    id: Mapped[uuid_primary_key]
    seq: Mapped[insert_sequence]
    idempotency_key: Mapped[str] = mapped_column(String(255))
    claim_assertion_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("claim_assertion.id", ondelete="RESTRICT")
    )
    poll_revision_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("poll_revision.id", ondelete="RESTRICT")
    )
    poll_average_revision_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            "poll_average_revision.id",
            ondelete="RESTRICT",
            name="fk_review_decision_poll_average_revision",
        )
    )
    outcome: Mapped[ReviewOutcome] = mapped_column(
        enum_type(ReviewOutcome, name="review_outcome")
    )
    reviewer_kind: Mapped[ReviewerKind] = mapped_column(
        enum_type(ReviewerKind, name="reviewer_kind")
    )
    reviewer_identifier: Mapped[str] = mapped_column(String(255))
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[created_at_timestamp]

    claim_assertion: Mapped[ClaimAssertion | None] = relationship(
        foreign_keys=[claim_assertion_id]
    )
    poll_revision: Mapped[PollRevision | None] = relationship(
        foreign_keys=[poll_revision_id]
    )
    poll_average_revision: Mapped[PollAverageRevision | None] = relationship(
        foreign_keys=[poll_average_revision_id]
    )

    __table_args__ = (
        UniqueConstraint(
            "idempotency_key",
            name="uq_review_decision_idempotency_key",
        ),
        CheckConstraint(
            """
            (
                claim_assertion_id IS NOT NULL
                AND poll_revision_id IS NULL
                AND poll_average_revision_id IS NULL
            )
            OR
            (
                claim_assertion_id IS NULL
                AND poll_revision_id IS NOT NULL
                AND poll_average_revision_id IS NULL
            )
            OR
            (
                claim_assertion_id IS NULL
                AND poll_revision_id IS NULL
                AND poll_average_revision_id IS NOT NULL
            )
            """,
            name="exactly_one_target",
        ),
        CheckConstraint(
            """
            outcome = 'accepted'
            OR (reason IS NOT NULL AND btrim(reason) <> '')
            """,
            name="nonacceptance_has_reason",
        ),
        CheckConstraint(
            "reviewer_identifier <> ''",
            name="reviewer_identifier_nonempty",
        ),
        UniqueConstraint("seq", name="uq_review_decision_seq"),
        Index(
            "ix_review_decision_claim_assertion_seq",
            "claim_assertion_id",
            "seq",
        ),
        Index(
            "ix_review_decision_poll_revision_seq",
            "poll_revision_id",
            "seq",
        ),
        Index(
            "ix_review_decision_poll_average_revision_seq",
            "poll_average_revision_id",
            "seq",
        ),
    )


class ReviewTask(Base):
    """Mutable queue state kept separate from immutable review decisions."""

    __tablename__ = "review_task"

    id: Mapped[uuid_primary_key]
    claim_assertion_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("claim_assertion.id", ondelete="RESTRICT")
    )
    poll_revision_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("poll_revision.id", ondelete="RESTRICT")
    )
    poll_average_revision_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("poll_average_revision.id", ondelete="RESTRICT")
    )
    created_by_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("research_run.id", ondelete="RESTRICT")
    )
    status: Mapped[ReviewTaskStatus] = mapped_column(
        enum_type(ReviewTaskStatus, name="review_task_status"),
        default=ReviewTaskStatus.PENDING,
        server_default=text(f"'{ReviewTaskStatus.PENDING.value}'"),
    )
    priority: Mapped[int] = mapped_column(
        Integer,
        default=DEFAULT_REVIEW_PRIORITY,
        server_default=text(str(DEFAULT_REVIEW_PRIORITY)),
    )
    assigned_to: Mapped[str | None] = mapped_column(String(255))
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[created_at_timestamp]
    completed_at: Mapped[utc_timestamp | None]

    claim_assertion: Mapped[ClaimAssertion | None] = relationship(
        foreign_keys=[claim_assertion_id]
    )
    poll_revision: Mapped[PollRevision | None] = relationship(
        foreign_keys=[poll_revision_id]
    )
    poll_average_revision: Mapped[PollAverageRevision | None] = relationship(
        foreign_keys=[poll_average_revision_id]
    )
    created_by_run: Mapped[ResearchRun | None] = relationship(
        foreign_keys=[created_by_run_id]
    )

    __table_args__ = (
        CheckConstraint(
            """
            (
                claim_assertion_id IS NOT NULL
                AND poll_revision_id IS NULL
                AND poll_average_revision_id IS NULL
            )
            OR
            (
                claim_assertion_id IS NULL
                AND poll_revision_id IS NOT NULL
                AND poll_average_revision_id IS NULL
            )
            OR
            (
                claim_assertion_id IS NULL
                AND poll_revision_id IS NULL
                AND poll_average_revision_id IS NOT NULL
            )
            """,
            name="exactly_one_target",
        ),
        CheckConstraint(
            "priority >= 0 AND priority <= 100",
            name="priority_range",
        ),
        CheckConstraint(
            """
            (
                status IN ('pending', 'claimed')
                AND completed_at IS NULL
            )
            OR
            (
                status IN ('completed', 'cancelled')
                AND completed_at IS NOT NULL
            )
            """,
            name="completion_matches_status",
        ),
        Index(
            "uq_review_task_open_claim_assertion",
            "claim_assertion_id",
            unique=True,
            postgresql_where=text(
                "claim_assertion_id IS NOT NULL AND status IN ('pending', 'claimed')"
            ),
        ),
        Index(
            "uq_review_task_open_poll_revision",
            "poll_revision_id",
            unique=True,
            postgresql_where=text(
                "poll_revision_id IS NOT NULL AND status IN ('pending', 'claimed')"
            ),
        ),
        Index(
            "uq_review_task_open_poll_average_revision",
            "poll_average_revision_id",
            unique=True,
            postgresql_where=text(
                "poll_average_revision_id IS NOT NULL "
                "AND status IN ('pending', 'claimed')"
            ),
        ),
        Index("ix_review_task_status_priority", "status", "priority"),
        Index("ix_review_task_created_by_run_id", "created_by_run_id"),
    )
