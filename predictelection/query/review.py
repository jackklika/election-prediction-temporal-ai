"""What is unreviewed — the fourth question, and the one about the graph itself.

The other projections answer questions about elections. This one answers "how
much of what you are looking at has anyone checked", which is the question that
makes the other three honest: a poll timeline drawn from unreviewed data is not
wrong, but a reader should be able to find out.

Read-only, unlike everything in `sql.review_queue` that shares its subject. An
API can expose this on a dashboard without granting the ability to decide
anything, which is why it lives here rather than there.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection.sql.review import (
    ReviewDecision,
    ReviewOutcome,
    ReviewTask,
    ReviewTaskStatus,
)


@dataclass(frozen=True, slots=True)
class ReviewBacklog:
    """The queue in one row, for a header or a dashboard tile."""

    pending: int
    claimed: int
    completed: int
    decisions: int
    accepted: int
    rejected: int

    @property
    def open(self) -> int:
        """Everything still waiting on a person."""

        return self.pending + self.claimed

    @property
    def is_clear(self) -> bool:
        return self.open == 0


def unreviewed(session: Session) -> ReviewBacklog:
    """Counts by status, plus what the decisions so far concluded.

    Decisions are counted by their *latest* verdict per target, not by row: the
    table is append-only and a reviewer may reverse themselves, so counting rows
    would report both the mistake and its correction as findings.
    """

    by_status = {
        status: count
        for status, count in session.execute(
            select(ReviewTask.status, func.count(ReviewTask.id)).group_by(
                ReviewTask.status
            )
        )
    }

    latest = (
        select(
            func.coalesce(
                ReviewDecision.claim_assertion_id,
                ReviewDecision.poll_revision_id,
                ReviewDecision.poll_average_revision_id,
            ).label("target"),
            func.max(ReviewDecision.seq).label("seq"),
        )
        .group_by("target")
        .subquery()
    )
    outcomes = {
        outcome: count
        for outcome, count in session.execute(
            select(ReviewDecision.outcome, func.count(ReviewDecision.id))
            .join(latest, ReviewDecision.seq == latest.c.seq)
            .group_by(ReviewDecision.outcome)
        )
    }

    return ReviewBacklog(
        pending=by_status.get(ReviewTaskStatus.PENDING, 0),
        claimed=by_status.get(ReviewTaskStatus.CLAIMED, 0),
        completed=by_status.get(ReviewTaskStatus.COMPLETED, 0),
        decisions=sum(outcomes.values()),
        accepted=outcomes.get(ReviewOutcome.ACCEPTED, 0),
        rejected=outcomes.get(ReviewOutcome.REJECTED, 0),
    )
