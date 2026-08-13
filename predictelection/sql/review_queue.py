"""Reading the review queue, and recording what a reviewer decided.

Four ingestion paths file `ReviewTask` rows and, until this module, nothing read
them: the queue was a write-only table and human-in-the-loop review was a stated
goal with no way to do it. What was missing was never the schema — `ReviewTask`,
`ReviewDecision`, `ClaimSupersession` and `EntityRedirect` were all there — it
was a way to *see* a task and a way to *answer* it.

Two ideas shape the design:

- **A task is only reviewable if its target is legible.** A row saying "new
  pollster 'UC Berkeley' resembles existing: 'UC Berkeley IGS'" is useless
  without the poll it came from and the entity ids either name resolves to. So
  reading a task loads its target, and for a pollster concern it re-runs the same
  lookalike query that filed it — `find_lookalikes`, not a second copy.
- **Deciding is append-only, and the fix is separate from the verdict.** The
  decision records what a human concluded; a merge additionally writes an
  `EntityRedirect`. Both happen in the caller's transaction, so a CLI that dies
  between them leaves neither.

Nothing here mutates a claim or a poll revision. A correction to a claim is a new
claim plus a `ClaimSupersession` (see `new_claim_supersession`), which is a
separate motion from working the queue and is not needed by any task currently
filed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from predictelection.sql.claim import Claim, ClaimAssertion, EvidenceAnchor
from predictelection.sql.entity import (
    Entity,
    EntityKind,
    EntityRedirect,
    normalize_slug,
    resolve_entity,
)
from predictelection.sql.lookup import find_lookalikes
from predictelection.sql.polling import (
    Poll,
    PollEstimate,
    PollOption,
    PollQuestion,
    PollRevision,
    PollSample,
)
from predictelection.sql.predicate import get_predicate_spec_by_id
from predictelection.sql.review import (
    ReviewDecision,
    ReviewerKind,
    ReviewOutcome,
    ReviewTask,
    ReviewTaskStatus,
    new_review_decision,
)


DEFAULT_LIMIT = 50
"""Enough to hold a session's worth of queue in one screen."""


@dataclass(frozen=True, slots=True)
class MergeCandidate:
    """An entity a reviewer might merge the task's subject into."""

    entity_id: uuid.UUID
    name: str
    canonical_name: str


@dataclass(frozen=True, slots=True)
class Reading:
    """One revision's account of a poll, as a reviewer needs to compare them."""

    revision_id: uuid.UUID
    revision_number: int
    is_subject: bool
    """Whether this is the reading the task was filed about."""

    summary: str
    recorded_by: str


@dataclass(frozen=True, slots=True)
class ReviewTarget:
    """What the task is about, rendered for someone who has to judge it.

    One shape for every target kind rather than a union: the reviewer's question
    is always "what is this, where did it come from, and what would I be
    agreeing to", and a caller that had to branch per kind to ask that would put
    the same three lines in three places.
    """

    kind: str
    """`poll_revision`, `poll_average_revision` or `claim_assertion`."""

    target_id: uuid.UUID
    headline: str
    """One line naming the thing, e.g. the pollster and the contest."""

    details: tuple[tuple[str, str], ...] = ()
    """Label/value pairs, in the order they should be shown."""

    subject_entity_id: uuid.UUID | None = None
    """The entity a merge would move, when the task is about one."""

    candidates: tuple[MergeCandidate, ...] = ()
    """Merge targets, when the concern is that two entities are one."""

    readings: tuple[Reading, ...] = ()
    """Every revision of this poll, so two accounts of it can be compared.

    Carries the revision ids rather than only the rendered text, because
    accepting one reading is only half an answer: the others are still live, and
    saying so needs their ids.
    """

    @property
    def rivals(self) -> tuple[Reading, ...]:
        """The readings that are not the one under review."""

        return tuple(reading for reading in self.readings if not reading.is_subject)


@dataclass(frozen=True, slots=True)
class ReviewTaskView:
    task_id: uuid.UUID
    status: ReviewTaskStatus
    priority: int
    reason: str | None
    created_at: datetime
    target: ReviewTarget
    created_by_run_id: uuid.UUID | None = None

    @property
    def short_id(self) -> str:
        """The prefix a reviewer types. Eight hex digits, like a git short SHA."""

        return str(self.task_id)[:8]


def pending_tasks(
    session: Session,
    *,
    status: ReviewTaskStatus | None = ReviewTaskStatus.PENDING,
    limit: int = DEFAULT_LIMIT,
) -> tuple[ReviewTaskView, ...]:
    """The queue, most urgent first.

    Priority ascending because `DEFAULT_REVIEW_PRIORITY` is the midpoint of the
    allowed range and lower is more urgent; created_at breaks ties so the oldest
    unanswered question is not permanently behind the newest.
    """

    statement = select(ReviewTask).order_by(ReviewTask.priority, ReviewTask.created_at)
    if status is not None:
        statement = statement.where(ReviewTask.status == status)
    tasks = session.scalars(statement.limit(limit)).all()
    return tuple(_view(session, task) for task in tasks)


def find_task(session: Session, prefix: str) -> ReviewTask:
    """One task by id or by a unique prefix of it.

    Prefixes because a queue addressed only by full UUID is not one a person can
    work: every action would start with a copy and paste. An ambiguous prefix
    raises rather than picking, and so does one that matches nothing.
    """

    cleaned = prefix.strip().lower()
    if not cleaned:
        raise ValueError("give a task id or a prefix of one")

    try:
        return _require(session, uuid.UUID(cleaned))
    except ValueError:
        pass

    matches = [
        task
        for task in session.scalars(select(ReviewTask))
        if str(task.id).startswith(cleaned)
    ]
    if not matches:
        raise ValueError(f"no review task starts with {prefix!r}")
    if len(matches) > 1:
        found = ", ".join(str(task.id)[:12] for task in matches[:5])
        raise ValueError(f"{prefix!r} matches {len(matches)} tasks: {found}")
    return matches[0]


def task_view(session: Session, task: ReviewTask) -> ReviewTaskView:
    """The full picture of one task, target included."""

    return _view(session, task)


def decide(
    session: Session,
    task: ReviewTask,
    *,
    outcome: ReviewOutcome,
    reviewer: str,
    action_token: str,
    reason: str | None = None,
    reviewer_kind: ReviewerKind = ReviewerKind.HUMAN,
) -> ReviewDecision:
    """Record a verdict and close the task.

    The two writes belong together: a decision with the task left pending would
    show up again tomorrow, and a closed task with no decision would lose why.
    `ck_review_task_completion_matches_status` enforces the second half of it —
    a completed task must carry a `completed_at`.

    Acceptance normally needs no argument, with one exception: a task that offers
    merge candidates. Accepting *that* is a substantive finding — "I looked at
    these two and they are different things" — and it is the answer that leaves no
    other trace. A merge writes a redirect anyone can see later; declining one
    writes nothing at all, so if the reason is blank the reasoning is gone, and a
    reader cannot tell a judgement from a stray keypress.
    """

    if task.status in {ReviewTaskStatus.COMPLETED, ReviewTaskStatus.CANCELLED}:
        raise ValueError(f"task {task.id} is already {task.status.value}")

    if (
        outcome is ReviewOutcome.ACCEPTED
        and not (reason or "").strip()
        and _target(session, task).candidates
    ):
        raise ValueError(
            "this task offers merge candidates, so accepting it without merging "
            "must say why they are not the same thing"
        )

    decision = new_review_decision(
        action_token=action_token,
        outcome=outcome,
        reviewer_identifier=reviewer,
        reviewer_kind=reviewer_kind,
        reason=reason,
        claim_assertion_id=task.claim_assertion_id,
        poll_revision_id=task.poll_revision_id,
        poll_average_revision_id=task.poll_average_revision_id,
    )
    session.add(decision)

    task.status = ReviewTaskStatus.COMPLETED
    task.completed_at = datetime.now(UTC)
    session.flush()
    return decision


def decide_reading(
    session: Session,
    revision_id: uuid.UUID,
    *,
    outcome: ReviewOutcome,
    reviewer: str,
    action_token: str,
    reason: str,
    reviewer_kind: ReviewerKind = ReviewerKind.HUMAN,
) -> ReviewDecision:
    """Record a verdict on a poll revision that has no task of its own.

    Accepting one reading of a poll is only half an answer. The rival reading is
    still there, still has its own `poll_option` and `poll_estimate` rows, and any
    query that does not filter by revision still sees it — so "revision 2 is
    right" has to be sayable as "and revision 1 is not".

    A separate function from `decide` because there is no queue state to move:
    only the task-bearing revision was ever queued. And it cannot be done by
    editing the losing revision either — `PollRevision` is immutable, so
    `supersedes_revision_id` can only be set when the row is written, which is
    the importer's moment and not the reviewer's. The decision *is* the record.

    A reason is required in every direction here, including acceptance: a verdict
    on a row nobody asked about needs to say what prompted it.
    """

    if not reason.strip():
        raise ValueError("a verdict on an unqueued reading must say why")
    if session.get(PollRevision, revision_id) is None:
        raise ValueError(f"no poll revision {revision_id}")

    open_task = session.scalar(
        select(ReviewTask.id).where(
            ReviewTask.poll_revision_id == revision_id,
            ReviewTask.status.in_({ReviewTaskStatus.PENDING, ReviewTaskStatus.CLAIMED}),
        )
    )
    if open_task is not None:
        # Deciding it here would leave its task pending and it would come back
        # tomorrow, with a decision already recorded against it.
        raise ValueError(
            f"revision {revision_id} has open task {str(open_task)[:8]}; "
            "decide that instead"
        )

    decision = new_review_decision(
        action_token=action_token,
        outcome=outcome,
        reviewer_identifier=reviewer,
        reviewer_kind=reviewer_kind,
        reason=reason,
        poll_revision_id=revision_id,
    )
    session.add(decision)
    session.flush()
    return decision


def merge_entities(
    session: Session,
    *,
    duplicate_id: uuid.UUID,
    canonical_id: uuid.UUID,
    reviewer: str,
    reason: str,
) -> EntityRedirect:
    """Declare that one entity was a duplicate of another.

    The redirect is written against `resolve_entity(canonical_id)` rather than
    against whatever the reviewer typed. Merging B into A and later A into C
    would otherwise leave B -> A -> C, and every read would depend on walking the
    chain correctly; pointing B straight at C keeps every redirect one hop.

    Refuses the two ways a merge can corrupt the graph rather than trusting the
    caller: merging something into itself, and merging in the direction that
    would make a cycle. `resolve_entity` guards its own walk against cycles, but
    a cycle here means neither entity has a canonical form at all.
    """

    if not reason.strip():
        raise ValueError("a merge must say why")
    if session.get(Entity, duplicate_id) is None:
        raise ValueError(f"no entity {duplicate_id}")
    if session.get(Entity, canonical_id) is None:
        raise ValueError(f"no entity {canonical_id}")
    if session.get(EntityRedirect, duplicate_id) is not None:
        raise ValueError(f"{duplicate_id} has already been merged into something")

    canonical_id = resolve_entity(session, canonical_id)
    if canonical_id == duplicate_id:
        # Either the reviewer named the same entity twice, or the canonical they
        # named already redirects back here. Both would make a cycle, and a cycle
        # means neither entity has a canonical form at all.
        raise ValueError("an entity cannot be a duplicate of itself")

    redirect = EntityRedirect(
        duplicate_entity_id=duplicate_id,
        canonical_entity_id=canonical_id,
        reason=reason.strip(),
        created_by=reviewer.strip(),
    )
    session.add(redirect)
    session.flush()
    return redirect


# --------------------------------------------------------------------------


def _require(session: Session, task_id: uuid.UUID) -> ReviewTask:
    task = session.get(ReviewTask, task_id)
    if task is None:
        raise ValueError(f"no review task {task_id}")
    return task


def _view(session: Session, task: ReviewTask) -> ReviewTaskView:
    return ReviewTaskView(
        task_id=task.id,
        status=task.status,
        priority=task.priority,
        reason=task.reason,
        created_at=task.created_at,
        created_by_run_id=task.created_by_run_id,
        target=_target(session, task),
    )


def _target(session: Session, task: ReviewTask) -> ReviewTarget:
    if task.poll_revision_id is not None:
        return _poll_revision_target(session, task.poll_revision_id)
    if task.claim_assertion_id is not None:
        return _claim_assertion_target(session, task.claim_assertion_id)
    if task.poll_average_revision_id is not None:
        return ReviewTarget(
            kind="poll_average_revision",
            target_id=task.poll_average_revision_id,
            headline="poll average revision",
        )
    raise ValueError(f"task {task.id} has no target")  # pragma: no cover


def _poll_revision_target(session: Session, revision_id: uuid.UUID) -> ReviewTarget:
    """A poll revision with the pollster a merge would move.

    The lookalikes come from `find_lookalikes`, the same query that filed the
    task. Re-derived rather than stored because the answer changes as the graph
    does: an organization created after the task was filed is a legitimate merge
    target for it, and a stored list would never mention it.
    """

    revision = session.get(PollRevision, revision_id)
    if revision is None:
        raise ValueError(f"no poll revision {revision_id}")  # pragma: no cover

    pollster = session.get(Entity, revision.pollster_id)
    pollster_name = pollster.canonical_name if pollster else "(unknown pollster)"
    contest = _contest_of(session, revision_id)

    details: list[tuple[str, str]] = [
        ("pollster", f"{pollster_name}  [{str(revision.pollster_id)[:8]}]"),
        ("contest", contest or "(none recorded)"),
        ("fieldwork", _fieldwork(revision)),
        ("revision", str(revision.revision_number)),
        ("recorded by", revision.created_by or "(unattributed)"),
    ]

    poll = session.get(Poll, revision.poll_id)
    if poll is not None and poll.external_id:
        details.append(("poll key", poll.external_id))

    # Every reading of this poll, this one included. A task filed because "two
    # sources disagree about its contents" is undecidable without them: the
    # payload itself is not stored, only its hash, so the disagreement lives in
    # these rows and nowhere else.
    readings = tuple(
        Reading(
            revision_id=sibling.id,
            revision_number=sibling.revision_number,
            is_subject=sibling.id == revision_id,
            summary=_readings(session, sibling.id),
            recorded_by=sibling.created_by or "(unattributed)",
        )
        for sibling in _revisions_of(session, revision.poll_id)
    )

    return ReviewTarget(
        kind="poll_revision",
        target_id=revision_id,
        headline=f"{pollster_name} — {contest or 'unknown contest'}",
        details=tuple(details),
        subject_entity_id=revision.pollster_id,
        candidates=_merge_candidates(session, pollster_name, revision.pollster_id),
        readings=readings,
    )


def _merge_candidates(
    session: Session, name: str, exclude: uuid.UUID | None
) -> tuple[MergeCandidate, ...]:
    """Organizations this pollster might be a duplicate of.

    The same `find_lookalikes` call the ingestor made when it filed the task, at
    the same threshold. Two copies of the query would let the reviewer be offered
    a candidate the ingestor never considered — the diagnosis and the fix
    disagreeing about the same graph.
    """

    slug = normalize_slug(name)
    seen: set[uuid.UUID] = {exclude} if exclude else set()
    candidates: list[MergeCandidate] = []
    for entity_id, alias in find_lookalikes(
        session, slug, kind=EntityKind.ORGANIZATION
    ):
        canonical_id = resolve_entity(session, entity_id)
        if canonical_id in seen:
            continue
        entity = session.get(Entity, canonical_id)
        if entity is None or entity.kind is not EntityKind.ORGANIZATION:
            continue  # pragma: no cover - kind is filtered in the query already
        seen.add(canonical_id)
        candidates.append(
            MergeCandidate(
                entity_id=canonical_id,
                name=alias,
                canonical_name=entity.canonical_name,
            )
        )
    return tuple(candidates)


def _revisions_of(session: Session, poll_id: uuid.UUID) -> Sequence[PollRevision]:
    return session.scalars(
        select(PollRevision)
        .where(PollRevision.poll_id == poll_id)
        .order_by(PollRevision.revision_number)
    ).all()


def _readings(session: Session, revision_id: uuid.UUID) -> str:
    """One revision's numbers, as "LV: Schiff 20, Porter 17".

    Grouped by sample, because a revision can hold several — one row per
    published line, which is how a poll showing both likely and registered
    voters is stored. Flattened into one list they interleave, and two readings
    of the same poll become impossible to compare, which is the one thing this
    is for.

    Labels are verbatim: resolving "Garvey" to a person is a job for whoever
    knows the contest's candidates, and a reviewer wants the printed label anyway.
    """

    rows = session.execute(
        select(
            PollSample.label,
            PollSample.position,
            PollOption.label,
            PollEstimate.percentage,
        )
        .join(PollEstimate, PollEstimate.option_id == PollOption.id)
        .join(PollSample, PollSample.id == PollEstimate.sample_id)
        .where(PollOption.poll_revision_id == revision_id)
        .order_by(PollSample.position, PollOption.position)
    ).all()
    if not rows:
        return "(no readings)"

    samples: dict[tuple[int, str], list[str]] = {}
    for sample_label, position, option_label, percentage in rows:
        number = "?" if percentage is None else f"{float(percentage):g}"
        samples.setdefault((position, sample_label), []).append(
            f"{option_label} {number}"
        )
    # Two samples of one revision can carry the same label — Wikipedia prints
    # two LV lines when a poll was published with and without a candidate — so
    # the position disambiguates them once there is more than one.
    return "  |  ".join(
        f"{label if len(samples) == 1 else f'{position + 1}. {label}'}: "
        f"{', '.join(readings)}"
        for (position, label), readings in sorted(samples.items())
    )


def _contest_of(session: Session, revision_id: uuid.UUID) -> str | None:
    contest_id = session.scalar(
        select(PollQuestion.contest_id)
        .where(PollQuestion.poll_revision_id == revision_id)
        .limit(1)
    )
    if contest_id is None:
        return None
    contest = session.get(Entity, contest_id)
    return contest.canonical_name if contest else None


def _fieldwork(revision: PollRevision) -> str:
    return (
        f"{_day(revision.fieldwork_started_on)} to {_day(revision.fieldwork_ended_on)}"
    )


def _day(value: date | datetime | None) -> str:
    return value.isoformat() if value is not None else "?"


def _claim_assertion_target(session: Session, assertion_id: uuid.UUID) -> ReviewTarget:
    """A claim rendered as the sentence it asserts, plus what evidences it."""

    assertion = session.get(ClaimAssertion, assertion_id)
    if assertion is None:
        raise ValueError(f"no claim assertion {assertion_id}")  # pragma: no cover
    claim = session.get(Claim, assertion.claim_id)
    if claim is None:
        raise ValueError(f"no claim {assertion.claim_id}")  # pragma: no cover

    predicate = get_predicate_spec_by_id(claim.predicate_version_id).slug
    subject = _name(session, claim.subject_id)
    obj = _name(session, claim.object_id)
    headline = f"{subject} — {predicate} — {obj or claim.value}"

    details: list[tuple[str, str]] = [
        ("predicate", predicate),
        ("subject", f"{subject}  [{str(claim.subject_id)[:8]}]"),
    ]
    if claim.object_id is not None:
        details.append(("object", f"{obj}  [{str(claim.object_id)[:8]}]"))
    if claim.value:
        details.append(("value", str(claim.value)))
    details.append(("asserted by", assertion.asserted_by or "(unattributed)"))
    details.append(("stance", assertion.stance.value))

    anchor = session.get(EvidenceAnchor, assertion.evidence_anchor_id)
    if anchor is not None and anchor.excerpt:
        details.append(("evidence", anchor.excerpt[:200]))

    return ReviewTarget(
        kind="claim_assertion",
        target_id=assertion_id,
        headline=headline,
        details=tuple(details),
        subject_entity_id=claim.subject_id,
    )


def _name(session: Session, entity_id: uuid.UUID | None) -> str | None:
    if entity_id is None:
        return None
    entity = session.get(Entity, entity_id)
    return entity.canonical_name if entity else str(entity_id)


__all__ = [
    "DEFAULT_LIMIT",
    "MergeCandidate",
    "Reading",
    "ReviewTarget",
    "ReviewTaskView",
    "decide",
    "decide_reading",
    "find_task",
    "merge_entities",
    "pending_tasks",
    "task_view",
]
