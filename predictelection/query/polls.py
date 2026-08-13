"""Polling for a contest over time — the shape backtesting wants.

The one projection here that claims cannot serve: polls have nine tables of
their own and never enter the claim graph, because a poll is a published
measurement rather than an assertion about the world.

**The rule this file exists to enforce:** a poll can hold more than one revision
with different numbers, and which one to believe is a `ReviewDecision`, not a
column on the row. "Take the latest revision" is the obvious default and it is
wrong — the later revision is simply the one written second, and a reviewer who
looked at both may have accepted the first. So the timeline reads the review
decisions and drops what a reviewer rejected.

Unreviewed revisions are included. Absence of review is not a verdict, and a
timeline that showed only human-blessed polls would be empty on any contest
nobody has worked through — which is every contest today.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection.query.claims import EntityRef
from predictelection.sql.entity import Entity, resolve_entity
from predictelection.sql.polling import (
    PollEstimate,
    PollOption,
    PollQuestion,
    PollRevision,
    PollSample,
)
from predictelection.sql.review import ReviewDecision, ReviewOutcome


@dataclass(frozen=True, slots=True)
class PollReadingRow:
    """One published number, with the label exactly as the source printed it.

    `choice` is populated only when someone resolved the label to an entity;
    the Wikipedia importer deliberately does not, because guessing that "Rogers"
    is a particular person mints a PERSON from a table cell.
    """

    label: str
    percentage: Decimal | None
    choice: EntityRef | None = None


@dataclass(frozen=True, slots=True)
class PollPoint:
    """One sample of one poll: a dated set of readings.

    A *sample*, not a revision, because a revision can publish several — likely
    voters and registered voters, or with and without a candidate — and averaging
    them or picking one arbitrarily is how a timeline starts lying. Each becomes
    its own point, labelled, and the caller decides which series to draw.
    """

    revision_id: uuid.UUID
    pollster: EntityRef | None
    sample_label: str
    population: str | None
    sample_size: int | None
    margin_of_error: Decimal | None
    fieldwork_started_on: date | None
    fieldwork_ended_on: date | None
    readings: tuple[PollReadingRow, ...]
    reviewed: bool = False
    """Whether a human has ruled on this revision at all. Unreviewed is the
    normal state, not a warning."""


def poll_timeline(
    session: Session,
    contest_id: uuid.UUID,
    *,
    include_rejected: bool = False,
    limit: int = 500,
) -> tuple[PollPoint, ...]:
    """Every accepted reading of every poll of this contest, oldest first.

    Ordered by fieldwork end, which is when the measurement stopped being about
    the world and started being about the past. Undated polls sort last rather
    than being dropped: they cannot be placed on a timeline but they exist, and
    silently omitting them would misstate how much polling a race had.
    """

    revision_ids = list(
        session.scalars(
            select(PollQuestion.poll_revision_id)
            .where(PollQuestion.contest_id == contest_id)
            .distinct()
        )
    )
    if not revision_ids:
        return ()

    verdicts = _verdicts(session, revision_ids)
    if not include_rejected:
        revision_ids = [
            revision_id
            for revision_id in revision_ids
            if verdicts.get(revision_id) is not ReviewOutcome.REJECTED
        ]
        if not revision_ids:
            return ()

    revisions = {
        revision.id: revision
        for revision in session.scalars(
            select(PollRevision).where(PollRevision.id.in_(revision_ids))
        )
    }
    pollsters = _pollsters(session, list(revisions.values()))

    points: list[PollPoint] = []
    for revision_id, revision in revisions.items():
        for sample in session.scalars(
            select(PollSample)
            .where(PollSample.poll_revision_id == revision_id)
            .order_by(PollSample.position)
        ):
            points.append(
                PollPoint(
                    revision_id=revision_id,
                    pollster=pollsters.get(revision.pollster_id),
                    sample_label=sample.label,
                    population=sample.population,
                    sample_size=sample.sample_size,
                    margin_of_error=sample.margin_of_error,
                    fieldwork_started_on=revision.fieldwork_started_on,
                    fieldwork_ended_on=revision.fieldwork_ended_on,
                    readings=_readings(session, sample.id),
                    reviewed=revision_id in verdicts,
                )
            )

    points.sort(
        key=lambda point: (
            point.fieldwork_ended_on is None,
            point.fieldwork_ended_on or date.min,
            point.sample_label,
        )
    )
    return tuple(points[:limit])


# --------------------------------------------------------------------------


def _verdicts(
    session: Session, revision_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, ReviewOutcome]:
    """The current verdict per revision — highest seq wins.

    By `seq` rather than `created_at`: now() is evaluated once per transaction,
    so decisions written together are indistinguishable by time, which is the
    reason `ReviewDecision` carries an insert sequence at all.
    """

    latest = (
        select(
            ReviewDecision.poll_revision_id,
            func.max(ReviewDecision.seq).label("seq"),
        )
        .where(ReviewDecision.poll_revision_id.in_(revision_ids))
        .group_by(ReviewDecision.poll_revision_id)
        .subquery()
    )
    rows = session.execute(
        select(ReviewDecision.poll_revision_id, ReviewDecision.outcome).join(
            latest, ReviewDecision.seq == latest.c.seq
        )
    )
    return {revision_id: outcome for revision_id, outcome in rows}


def _pollsters(
    session: Session, revisions: Sequence[PollRevision]
) -> dict[uuid.UUID, EntityRef]:
    """Pollster refs, resolved through any merge a reviewer has recorded.

    Through `resolve_entity` because a revision written before a merge still
    stores the duplicate's id: the redirect is a read-time indirection, not a
    backfill. Without this the same firm appears twice in one timeline under
    two names, which is the exact duplicate the merge was meant to end.
    """

    resolved: dict[uuid.UUID, EntityRef] = {}
    for revision in revisions:
        stored = revision.pollster_id
        if stored is None or stored in resolved:
            continue
        entity = session.get(Entity, resolve_entity(session, stored))
        if entity is not None:
            resolved[stored] = EntityRef(
                entity_id=entity.id, kind=entity.kind, name=entity.canonical_name
            )
    return resolved


def _readings(session: Session, sample_id: uuid.UUID) -> tuple[PollReadingRow, ...]:
    rows = session.execute(
        select(PollOption.label, PollEstimate.percentage, PollOption.choice_entity_id)
        .join(PollEstimate, PollEstimate.option_id == PollOption.id)
        .where(PollEstimate.sample_id == sample_id)
        .order_by(PollOption.position)
    ).all()
    return tuple(
        PollReadingRow(label=label, percentage=percentage, choice=None)
        for label, percentage, _ in rows
    )
