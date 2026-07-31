"""Everything with a side effect: fetching, archiving, and writing claims.

Activities rather than workflow code because all of it is non-deterministic — a
workflow replays, and re-running an HTTP GET during replay would be both wrong
and slow. Each one is idempotent, so Temporal's at-least-once delivery lands on
at-most-once effects: re-fetching returns the same archived digest, and
re-ingesting returns the same assertions.

Defined as methods on a class so the engine and object store are injected once
rather than reached for through module globals, which is also what lets the
tests run them against the same fixtures everything else uses.

Deliberately sync (`def`, not `async def`): SQLAlchemy and boto3 are both
blocking here, and an async activity that blocks the event loop starves every
other task in the worker. Temporal runs these on the worker's thread pool.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from temporalio import activity
from temporalio.exceptions import ApplicationError

from predictelection.activities.contracts import (
    ArchiveUrlInput,
    ArchiveUrlOutput,
    EntityMatchOutput,
    FindEntitiesInput,
    FindEntitiesOutput,
    FindEventsInput,
    FinishResearchRunInput,
    FinishResearchRunOutput,
    IngestRecordInput,
    IngestRecordOutput,
    StartResearchRunInput,
    StartResearchRunOutput,
)
from predictelection.research.archive import SourceArchive
from predictelection.research.ingestion import IngestContext
from predictelection.research.registry import ingestor_for
from predictelection.sql import (
    ClaimOutcome,
    EntityMatches,
    ResearchRun,
    ResearchRunStatus,
    SourceSnapshot,
    find_entities,
    find_events,
    get_or_create,
    idempotency_key,
)
from predictelection.storage.base import ObjectStore


MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
"""Refuse to archive anything larger, so one bad URL cannot fill the bucket."""

FETCH_TIMEOUT_SECONDS = 30.0


class ResearchActivities:
    """Activity implementations bound to a database and an object store."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        store: ObjectStore,
        http: httpx.Client | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._store = store
        self._http = http or httpx.Client(
            timeout=FETCH_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={"User-Agent": "predictelection/0.1 (research archiver)"},
        )

    def all(self) -> list[Callable[..., Any]]:
        """Every activity, for worker registration."""

        return [
            self.start_research_run,
            self.finish_research_run,
            self.archive_url,
            self.ingest_record,
            self.find_entities,
            self.find_events,
        ]

    # ----------------------------------------------------------------------

    @activity.defn(name="start_research_run")
    def start_research_run(
        self, request: StartResearchRunInput
    ) -> StartResearchRunOutput:
        """Open the run that every claim from this workflow will be attributed to."""

        # Scoped to the execution, not just the subject: retries of this run share
        # a row, a later run of the same research gets its own. workflow_id and
        # workflow_run_id are stored too, which is what
        # uq_research_run_temporal_execution exists to protect.
        key = idempotency_key(
            "research_run",
            task_type=request.task_type,
            subject=request.subject,
            workflow_run_id=request.workflow_run_id,
        )
        with self._session_factory() as session, session.begin():
            run, created = get_or_create(
                session,
                ResearchRun(
                    idempotency_key=key,
                    task_type=request.task_type,
                    status=ResearchRunStatus.RUNNING,
                    workflow_id=request.workflow_id,
                    workflow_run_id=request.workflow_run_id,
                    agent_name=request.agent_name,
                    model_id=request.model_id,
                    input_data={"subject": request.subject},
                ),
                key=ResearchRun.idempotency_key == key,
            )
            return StartResearchRunOutput(
                research_run_id=run.id, already_running=not created
            )

    @activity.defn(name="finish_research_run")
    def finish_research_run(
        self, request: FinishResearchRunInput
    ) -> FinishResearchRunOutput:
        with self._session_factory() as session, session.begin():
            run = session.get(ResearchRun, request.research_run_id)
            if run is None:
                raise ApplicationError(
                    f"no research run {request.research_run_id}", non_retryable=True
                )
            run.status = request.status
            run.completed_at = datetime.now(UTC)
            # ck_research_run_status_matches_outcome insists a failure says why
            run.error_message = (
                request.error_message or "failed without a message"
                if request.status is ResearchRunStatus.FAILED
                else None
            )
            return FinishResearchRunOutput(
                research_run_id=run.id, status=request.status
            )

    @activity.defn(name="find_entities")
    def find_entities(self, request: FindEntitiesInput) -> FindEntitiesOutput:
        """Let the agent see what the graph already knows.

        Read-only, so it is safe to repeat and cannot corrupt a retried run. The
        canonical_name it returns is the point: an agent that echoes an existing
        title back stops forking the entity.
        """

        with self._session_factory() as session:
            return _as_output(
                find_entities(
                    session,
                    name=request.name,
                    kind=request.kind,
                    limit=request.limit,
                )
            )

    @activity.defn(name="find_events")
    def find_events(self, request: FindEventsInput) -> FindEntitiesOutput:
        """The same, for events, which carry filters no other kind has.

        A separate activity rather than a branch inside find_entities: the
        branch had to guess which query the caller meant, guessed from the date
        fields, and threw `kind` away when it guessed events.
        """

        with self._session_factory() as session:
            return _as_output(
                find_events(
                    session,
                    name=request.name,
                    participant_ids=request.participant_ids,
                    jurisdiction_id=request.jurisdiction_id,
                    event_kind=request.event_kind,
                    occurred_after=request.occurred_after,
                    occurred_before=request.occurred_before,
                    limit=request.limit,
                )
            )

    @activity.defn(name="archive_url")
    def archive_url(self, request: ArchiveUrlInput) -> ArchiveUrlOutput:
        """Fetch a page and keep the bytes, so claims can cite something fixed.

        A non-2xx or oversized response is non-retryable: retrying a 404 wastes
        the workflow's time and the answer will not change.
        """

        try:
            response = self._http.get(request.url)
        except httpx.HTTPError as error:
            # transport failures are worth retrying, so let Temporal do that
            raise ApplicationError(f"fetching {request.url} failed: {error}") from error

        if response.status_code >= 400:
            raise ApplicationError(
                f"{request.url} returned {response.status_code}", non_retryable=True
            )
        content = response.content
        if len(content) > MAX_ARCHIVE_BYTES:
            raise ApplicationError(
                f"{request.url} is {len(content)} bytes, over the archive limit",
                non_retryable=True,
            )

        media_type = response.headers.get("content-type", "").split(";")[0].strip()
        retrieved_at = datetime.now(UTC)

        with self._session_factory() as session, session.begin():
            archive = SourceArchive(session, self._store)
            snapshot = archive.observe(
                kind=request.kind,
                canonical_url=str(response.url),
                content=content,
                media_type=media_type or None,
                title=request.title,
                retrieved_at=retrieved_at,
            )
            artifact = snapshot.artifact
            # More than one snapshot on these bytes means an earlier fetch had
            # already archived them — a content check rather than a clock one,
            # since created_at is transaction time and would be ambiguous here.
            observations = session.scalar(
                select(func.count(SourceSnapshot.id)).where(
                    SourceSnapshot.artifact_id == artifact.id
                )
            )
            return ArchiveUrlOutput(
                source_snapshot_id=snapshot.id,
                source_id=snapshot.source_id,
                sha256=artifact.sha256,
                storage_uri=artifact.storage_uri,
                byte_length=artifact.byte_length,
                retrieved_at=snapshot.retrieved_at,
                already_archived=(observations or 0) > 1,
            )

    @activity.defn(name="ingest_record")
    def ingest_record(self, request: IngestRecordInput) -> IngestRecordOutput:
        """Turn one scraped record of any domain into entities and claims.

        Domain-free on purpose. The record's own type selects the ingestor, so
        adding a scrape does not touch this file — which is what stops the
        activity layer drifting once several domains write through it.
        """

        with self._session_factory() as session, session.begin():
            snapshot = session.get(SourceSnapshot, request.source_snapshot_id)
            if snapshot is None:
                raise ApplicationError(
                    f"no source snapshot {request.source_snapshot_id}",
                    non_retryable=True,
                )
            context = IngestContext(
                session=session,
                snapshot=snapshot,
                archive=SourceArchive(session, self._store),
                research_run_id=request.research_run_id,
                asserted_by=request.asserted_by,
            )
            result = ingestor_for(request.record)(request.record, context)
            session.flush()
            return IngestRecordOutput(
                subject_entity_id=result.subject_entity_id,
                subject_created=result.subject_created,
                related_entity_ids=result.related_entity_ids,
                assertion_ids=tuple(assertion.id for assertion in result.assertions),
                claims_created=result.count(ClaimOutcome.CREATED),
                claims_corroborated=result.count(ClaimOutcome.CORROBORATED),
                claims_unchanged=result.count(ClaimOutcome.UNCHANGED),
                misaligned_count=len(result.misaligned),
            )


def _as_output(found: EntityMatches) -> FindEntitiesOutput:
    """One output shape for both lookups, truncation flag included.

    Dropping `truncated` here would put the cap back where it started: invisible
    to the workflow, and therefore invisible to the model.
    """

    return FindEntitiesOutput(
        matches=tuple(
            EntityMatchOutput(
                entity_id=match.entity_id,
                kind=match.kind,
                canonical_name=match.canonical_name,
                aliases=match.aliases,
                occurred_at=match.occurred_at,
            )
            for match in found
        ),
        truncated=found.truncated,
    )


def build_activities(
    *,
    session_factory: sessionmaker[Session] | None = None,
    store: ObjectStore | None = None,
) -> ResearchActivities:
    """Wire the activities to real infrastructure, for the worker entrypoint."""

    if session_factory is None:
        from predictelection.clients.sqlalchemy_engine import SqlAlchemyEngineClient

        session_factory = SqlAlchemyEngineClient().session_factory
    if store is None:
        from predictelection.storage import S3Config, S3ObjectStore

        store = S3ObjectStore(S3Config())
    return ResearchActivities(session_factory=session_factory, store=store)


__all__ = [
    "MAX_ARCHIVE_BYTES",
    "ResearchActivities",
    "build_activities",
]
