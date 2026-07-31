"""Activities against real infrastructure, with HTTP stubbed.

The activities are the only place side effects happen, so these assert the two
properties Temporal depends on: every one is idempotent under retry, and a
permanent failure is marked non-retryable instead of being attempted forever.

HTTP is stubbed because the network is the one dependency whose behaviour we are
not testing; the database and object store are real.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from temporalio.exceptions import ApplicationError

from predictelection.activities.contracts import (
    ArchiveUrlInput,
    FinishResearchRunInput,
    IngestDebateInput,
    StartResearchRunInput,
)
from predictelection.activities.research import MAX_ARCHIVE_BYTES, ResearchActivities
from predictelection.research.debates import ScrapedDebate
from predictelection.research.scraped import ScrapedEntity
from predictelection.sql import (
    Artifact,
    Claim,
    ClaimAssertion,
    ResearchRun,
    ResearchRunStatus,
    SourceKind,
    SourceSnapshot,
    TimePrecision,
)


pytestmark = pytest.mark.postgres


PAGE = b"<html><body><h1>Michigan governor debate</h1></body></html>"


def _stub_http(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=PAGE, headers={"content-type": "text/html"})


@pytest.fixture
def activity_sessions(session: Session) -> sessionmaker[Session]:
    """Real sessions on the test's connection, so activities can commit.

    An activity opens its own transaction, which it must: in production each one
    commits independently. Binding to the same connection with
    join_transaction_mode="create_savepoint" turns those commits into savepoint
    releases, visible to the test's own session and still undone by the outer
    rollback in conftest.
    """

    return sessionmaker(
        bind=session.get_bind(),
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
    )


@pytest.fixture
def activities(
    activity_sessions: sessionmaker[Session], object_store
) -> ResearchActivities:
    return ResearchActivities(
        session_factory=activity_sessions,
        store=object_store,
        http=_stub_http(_ok),
    )


def _debate(**overrides: Any) -> ScrapedDebate:
    base: dict[str, Any] = {
        "title": "2026 Michigan Gubernatorial Debate",
        "source_url": "https://example.test/mi-debate",
        "starts_at": datetime(2026, 9, 15, 21, 0, tzinfo=UTC),
        "starts_at_precision": TimePrecision.MINUTE,
        "participants": (ScrapedEntity(name="Abdul El-Sayed"),),
        "jurisdiction": ScrapedEntity(name="Michigan"),
    }
    return ScrapedDebate(**(base | overrides))


# --------------------------------------------------------------------------


def test_starting_a_run_twice_reuses_it(
    activities: ResearchActivities, session: Session
) -> None:
    """A retried activity must not open a second run for the same research."""

    request = StartResearchRunInput(task_type="find_debates", subject="El-Sayed")
    first = activities.start_research_run(request)
    second = activities.start_research_run(request)

    assert first.research_run_id == second.research_run_id
    assert first.already_running is False
    assert second.already_running is True
    assert session.scalar(select(func.count(ResearchRun.id))) == 1


def test_a_different_subject_gets_its_own_run(
    activities: ResearchActivities, session: Session
) -> None:
    one = activities.start_research_run(
        StartResearchRunInput(task_type="find_debates", subject="El-Sayed")
    )
    two = activities.start_research_run(
        StartResearchRunInput(task_type="find_debates", subject="Whitmer")
    )
    assert one.research_run_id != two.research_run_id


def test_finishing_a_failed_run_always_records_a_reason(
    activities: ResearchActivities, session: Session
) -> None:
    """ck_research_run_status_matches_outcome rejects a silent failure."""

    started = activities.start_research_run(
        StartResearchRunInput(task_type="find_debates", subject="El-Sayed")
    )
    activities.finish_research_run(
        FinishResearchRunInput(
            research_run_id=started.research_run_id,
            status=ResearchRunStatus.FAILED,
            error_message=None,
        )
    )
    run = session.get(ResearchRun, started.research_run_id)
    assert run is not None
    assert run.status is ResearchRunStatus.FAILED
    assert run.error_message  # supplied rather than left null
    assert run.completed_at is not None


def test_finishing_an_unknown_run_is_not_retryable(
    activities: ResearchActivities,
) -> None:
    import uuid

    with pytest.raises(ApplicationError) as error:
        activities.finish_research_run(
            FinishResearchRunInput(research_run_id=uuid.uuid4())
        )
    assert error.value.non_retryable is True


def test_archiving_stores_the_bytes_and_the_snapshot(
    activities: ResearchActivities, object_store, session: Session
) -> None:
    result = activities.archive_url(
        ArchiveUrlInput(url="https://example.test/mi-debate", kind=SourceKind.WEB_PAGE)
    )

    assert result.byte_length == len(PAGE)
    assert object_store.get(result.storage_uri) == PAGE
    assert result.already_archived is False
    snapshot = session.get(SourceSnapshot, result.source_snapshot_id)
    assert snapshot is not None
    assert snapshot.artifact.sha256 == result.sha256


def test_refetching_unchanged_content_reuses_the_artifact(
    activities: ResearchActivities, session: Session
) -> None:
    """Same bytes, new observation: one artifact, two snapshots."""

    first = activities.archive_url(ArchiveUrlInput(url="https://example.test/a"))
    second = activities.archive_url(ArchiveUrlInput(url="https://example.test/a"))

    assert first.sha256 == second.sha256
    assert second.already_archived is True
    assert session.scalar(select(func.count(Artifact.id))) == 1


def test_a_404_is_not_retried(activity_sessions, object_store) -> None:
    activities = ResearchActivities(
        session_factory=activity_sessions,
        store=object_store,
        http=_stub_http(lambda request: httpx.Response(404)),
    )
    with pytest.raises(ApplicationError) as error:
        activities.archive_url(ArchiveUrlInput(url="https://example.test/gone"))
    assert error.value.non_retryable is True


def test_a_transport_error_stays_retryable(activity_sessions, object_store) -> None:
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    activities = ResearchActivities(
        session_factory=activity_sessions,
        store=object_store,
        http=_stub_http(explode),
    )
    with pytest.raises(ApplicationError) as error:
        activities.archive_url(ArchiveUrlInput(url="https://example.test/flaky"))
    assert error.value.non_retryable is not True


def test_an_oversized_response_is_refused(activity_sessions, object_store) -> None:
    big = b"x" * (MAX_ARCHIVE_BYTES + 1)
    activities = ResearchActivities(
        session_factory=activity_sessions,
        store=object_store,
        http=_stub_http(lambda request: httpx.Response(200, content=big)),
    )
    with pytest.raises(ApplicationError) as error:
        activities.archive_url(ArchiveUrlInput(url="https://example.test/huge"))
    assert error.value.non_retryable is True


def test_ingesting_is_idempotent_under_retry(
    activities: ResearchActivities, session: Session
) -> None:
    started = activities.start_research_run(
        StartResearchRunInput(task_type="find_debates", subject="El-Sayed")
    )
    archived = activities.archive_url(
        ArchiveUrlInput(url="https://example.test/mi-debate")
    )
    request = IngestDebateInput(
        debate=_debate(),
        source_snapshot_id=archived.source_snapshot_id,
        research_run_id=started.research_run_id,
    )

    first = activities.ingest_debate(request)
    claims_after_first = session.scalar(select(func.count(Claim.id)))
    second = activities.ingest_debate(request)

    assert first.event_id == second.event_id
    assert first.assertion_ids == second.assertion_ids
    assert session.scalar(select(func.count(Claim.id))) == claims_after_first
    assert first.misaligned_count == 0


def test_ingesting_against_a_missing_snapshot_is_not_retryable(
    activities: ResearchActivities,
) -> None:
    import uuid

    with pytest.raises(ApplicationError) as error:
        activities.ingest_debate(
            IngestDebateInput(debate=_debate(), source_snapshot_id=uuid.uuid4())
        )
    assert error.value.non_retryable is True


def test_claims_are_attributed_to_the_run_and_the_page(
    activities: ResearchActivities, session: Session
) -> None:
    started = activities.start_research_run(
        StartResearchRunInput(task_type="find_debates", subject="El-Sayed")
    )
    archived = activities.archive_url(
        ArchiveUrlInput(url="https://example.test/mi-debate")
    )
    result = activities.ingest_debate(
        IngestDebateInput(
            debate=_debate(),
            source_snapshot_id=archived.source_snapshot_id,
            research_run_id=started.research_run_id,
            asserted_by="find_debates",
        )
    )

    assertions = session.scalars(
        select(ClaimAssertion).where(ClaimAssertion.id.in_(result.assertion_ids))
    ).all()
    assert len(assertions) == len(result.assertion_ids)
    for assertion in assertions:
        assert assertion.research_run_id == started.research_run_id
        assert assertion.asserted_by == "find_debates"
        assert (
            assertion.evidence_anchor.source_snapshot_id == archived.source_snapshot_id
        )
