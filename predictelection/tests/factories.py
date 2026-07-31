"""Minimal valid rows for the PostgreSQL tests.

Every factory flushes, because the primary keys use a client-side uuid4 default
that SQLAlchemy only applies at INSERT time — an unflushed row has id None and
cannot be referenced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import count
import uuid

from sqlalchemy.orm import Session

from predictelection.sql import (
    Artifact,
    Entity,
    EntityKind,
    EvidenceAnchor,
    FullSourceLocator,
    Poll,
    PollQuestion,
    PollSample,
    PredicateSpec,
    ResearchRun,
    ResearchRunStatus,
    Source,
    SourceKind,
    SourceSnapshot,
    new_evidence_anchor,
    new_poll_revision,
)


_counter = count(1)


def unique(prefix: str = "") -> str:
    """A short distinct string, so unique constraints only fail on purpose."""

    return f"{prefix}{next(_counter)}"


def unique_sha256() -> str:
    return f"{next(_counter):064x}"


def make_entity(
    session: Session,
    *,
    kind: EntityKind = EntityKind.PERSON,
    canonical_name: str | None = None,
) -> Entity:
    entity = Entity(
        kind=kind,
        canonical_name=canonical_name or unique("entity-"),
    )
    session.add(entity)
    session.flush()
    return entity


def make_source(session, *, kind: SourceKind = SourceKind.WEB_PAGE) -> Source:
    source = Source(kind=kind, canonical_url=f"https://example.test/{unique()}")
    session.add(source)
    session.flush()
    return source


def make_artifact(session) -> Artifact:
    artifact = Artifact(
        sha256=unique_sha256(),
        storage_uri=f"s3://bucket/{unique()}",
        byte_length=1024,
    )
    session.add(artifact)
    session.flush()
    return artifact


def make_snapshot(
    session: Session,
    *,
    source: Source | None = None,
    artifact: Artifact | None = None,
    retrieved_at: datetime | None = None,
) -> SourceSnapshot:
    snapshot = SourceSnapshot(
        source_id=(source or make_source(session)).id,
        artifact_id=(artifact or make_artifact(session)).id,
        retrieved_at=retrieved_at or datetime(2026, 7, 30, 12, 0, tzinfo=UTC),
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def make_anchor(
    session: Session,
    *,
    snapshot: SourceSnapshot | None = None,
    excerpt: str | None = None,
) -> EvidenceAnchor:
    anchor = new_evidence_anchor(
        source_snapshot_id=(snapshot or make_snapshot(session)).id,
        locator=FullSourceLocator(),
        excerpt=excerpt or unique("excerpt-"),
    )
    session.add(anchor)
    session.flush()
    return anchor


def make_research_run(
    session: Session,
    *,
    status: ResearchRunStatus = ResearchRunStatus.RUNNING,
) -> ResearchRun:
    run = ResearchRun(
        idempotency_key=unique("run-"),
        task_type="find_debates",
        status=status,
        # pinned rather than server-defaulted, so completed_at can be ordered
        # after it without depending on the wall clock
        started_at=datetime(2026, 7, 30, 12, 0, tzinfo=UTC),
        completed_at=(
            None
            if status is ResearchRunStatus.RUNNING
            else datetime(2026, 7, 30, 13, 0, tzinfo=UTC)
        ),
        error_message="boom" if status is ResearchRunStatus.FAILED else None,
    )
    session.add(run)
    session.flush()
    return run


def make_poll_revision(
    session: Session,
    *,
    poll: Poll | None = None,
    revision_number: int = 1,
    payload: dict[str, object] | None = None,
):
    if poll is None:
        poll = Poll()
        session.add(poll)
        session.flush()
    revision = new_poll_revision(
        payload=payload or {"poll": unique()},
        poll_id=poll.id,
        revision_number=revision_number,
        source_snapshot_id=make_snapshot(session).id,
        origin="model",
    )
    session.add(revision)
    session.flush()
    return revision


def make_poll_sample(session: Session, *, revision, position: int = 0) -> PollSample:
    sample = PollSample(
        poll_revision_id=revision.id,
        position=position,
        label="Likely voters",
        population="likely_voters",
        sample_size=800,
    )
    session.add(sample)
    session.flush()
    return sample


def make_poll_question(
    session: Session, *, revision, position: int = 0
) -> PollQuestion:
    question = PollQuestion(
        poll_revision_id=revision.id,
        position=position,
        text="Who would you vote for?",
    )
    session.add(question)
    session.flush()
    return question


def make_claim_subject_and_object(
    session: Session,
    predicate: PredicateSpec,
) -> tuple[uuid.UUID, uuid.UUID | None]:
    """Entities whose kinds satisfy the predicate's declared domain."""

    subject = make_entity(session, kind=predicate.subject_kinds[0])
    if not predicate.object_kinds:
        return subject.id, None
    obj = make_entity(session, kind=predicate.object_kinds[0])
    return subject.id, obj.id
