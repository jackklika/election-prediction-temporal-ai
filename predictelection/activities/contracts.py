"""Pydantic inputs and outputs for every workflow and activity.

One model in, one model out, always — never bare positional arguments. Temporal
persists these in workflow history, so a running workflow will replay against
whatever the code says today: adding an optional field is compatible, but
changing a positional signature silently breaks in-flight executions. A single
model also makes the compatible change (a new field with a default) the easy one.

They live apart from both the agent and the activity implementations so a
workflow can import the contracts without dragging in an LLM client or a
database connection.
"""

from __future__ import annotations

from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field

from predictelection.research.registry import ScrapedPayload
from predictelection.sql import (
    EntityKind,
    PoliticalEventKind,
    ResearchRunStatus,
    SourceKind,
)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# Research run lifecycle
# --------------------------------------------------------------------------


class StartResearchRunInput(Contract):
    task_type: str = Field(max_length=100)
    subject: str = Field(description="What was researched.")
    workflow_id: str | None = None
    workflow_run_id: str | None = Field(
        default=None,
        description=(
            "Scopes the run to one execution. Without it the key would be "
            "{task_type, subject}, which yields a single run row per subject "
            "forever: two months of research collapse into one row, and a "
            "per-run alignment score averages across all of it. Retries of the "
            "same execution still share a run, which is the point."
        ),
    )
    agent_name: str | None = None
    model_id: str | None = None


class StartResearchRunOutput(Contract):
    research_run_id: uuid.UUID
    already_running: bool = False
    """True when an earlier attempt already opened this run."""


class FinishResearchRunInput(Contract):
    research_run_id: uuid.UUID
    status: ResearchRunStatus = ResearchRunStatus.SUCCEEDED
    error_message: str | None = None


class FinishResearchRunOutput(Contract):
    research_run_id: uuid.UUID
    status: ResearchRunStatus


# --------------------------------------------------------------------------
# Archiving
# --------------------------------------------------------------------------


class ArchiveUrlInput(Contract):
    url: str
    kind: SourceKind = SourceKind.WEB_PAGE
    title: str | None = None


class ArchiveUrlOutput(Contract):
    source_snapshot_id: uuid.UUID
    source_id: uuid.UUID
    sha256: str = Field(min_length=64, max_length=64)
    storage_uri: str
    byte_length: int
    retrieved_at: datetime
    already_archived: bool
    """True when these exact bytes were already stored from an earlier fetch."""


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


class IngestRecordInput(Contract):
    """One scraped record of any domain, plus the evidence it must cite.

    Generic rather than one model per domain: the record itself is the only part
    that differs, and `ScrapedPayload` already carries which kind it is. A
    per-domain pair meant every new scrape edited this file, the activity, the
    workflow and the worker — four structural changes for one domain.
    """

    record: ScrapedPayload
    source_snapshot_id: uuid.UUID
    research_run_id: uuid.UUID | None = None
    asserted_by: str | None = None


class IngestRecordOutput(Contract):
    """What one record put in the graph, in terms no domain owns.

    `subject_entity_id` is whatever the record was chiefly about — a debate, a
    candidacy, an endorsement. Naming it `event_id` made the counters below,
    which are identical for every domain, look debate-specific.
    """

    subject_entity_id: uuid.UUID
    subject_created: bool = False
    """False means the subject was already known under this identity."""

    related_entity_ids: tuple[uuid.UUID, ...] = ()
    assertion_ids: tuple[uuid.UUID, ...] = ()
    claims_created: int = 0
    """Propositions nothing had asserted before."""

    claims_corroborated: int = 0
    """Known propositions, now backed by further evidence."""

    claims_unchanged: int = 0
    """Already asserted from this evidence by this run — a retry wrote nothing."""

    misaligned_count: int = 0
    """Assertions whose entity kinds did not match the predicate's domain.

    They are stored and queued for review rather than dropped, so this is a
    quality signal about the extraction, not a failure count.
    """


# --------------------------------------------------------------------------
# Lookup — what the agent reads before it writes
# --------------------------------------------------------------------------


class FindEntitiesInput(Contract):
    name: str | None = Field(
        default=None, description="Name or fragment to search for."
    )
    kind: EntityKind | None = Field(
        default=None, description="Narrow to one kind, e.g. person or contest."
    )
    limit: int = Field(default=20, ge=1, le=100)


class FindEventsInput(Contract):
    """Events specifically, which need filters no other kind has.

    Separate from FindEntitiesInput rather than a superset of it because the
    single-activity version had to *infer* which query to run, and inferred it
    from the date fields: asking for people within a date range silently
    searched events instead, discarding `kind` entirely.
    """

    name: str | None = Field(
        default=None, description="Name or fragment to search for."
    )
    participant_ids: tuple[uuid.UUID, ...] = Field(
        default=(),
        description=(
            "Only events these entities took part in. Scopes the result to the "
            "subject being researched instead of the graph as a whole. Plural "
            "because a name can resolve to more than one person."
        ),
    )
    jurisdiction_id: uuid.UUID | None = Field(
        default=None, description="Only events held in this jurisdiction."
    )
    event_kind: PoliticalEventKind | None = Field(
        default=None, description="Only events of this kind, e.g. debate."
    )
    occurred_after: datetime | None = Field(
        default=None, description="Earliest date to consider."
    )
    occurred_before: datetime | None = Field(
        default=None, description="Latest date to consider."
    )
    limit: int = Field(default=20, ge=1, le=100)


class EntityMatchOutput(Contract):
    entity_id: uuid.UUID
    kind: EntityKind
    canonical_name: str
    aliases: tuple[str, ...] = ()
    occurred_at: datetime | None = None


class FindEntitiesOutput(Contract):
    matches: tuple[EntityMatchOutput, ...] = ()
    truncated: bool = False
    """True when the limit hid further matches.

    Callers must surface this. A capped list rendered as if it were complete
    tells the model nothing else exists, so it re-describes what it cannot see —
    which is the duplicate this lookup exists to prevent.
    """


# --------------------------------------------------------------------------
# Workflow
# --------------------------------------------------------------------------


class ResearchInput(Contract):
    """What every agent-driven research workflow is asked for.

    One model for all of them: a workflow that needed a different question would
    be asking the agent something the scaffold cannot drive anyway.
    """

    subject: str = Field(
        min_length=1,
        description="Politician or race to research, e.g. 'Abdul El-Sayed'.",
    )
    asserted_by: str | None = None


class ResearchOutput(Contract):
    """What one research run learned, counted the same way for every domain.

    "records" rather than "debates" because the loop that fills these in does
    not know what it is looping over — which is the point.
    """

    research_run_id: uuid.UUID
    records_found: int = 0
    records_new: int = 0
    """Subjects not already in the graph. The number that actually matters."""

    records_already_known: int = 0
    """Re-discoveries. A run of all re-discoveries has learned nothing."""

    claims_created: int = 0
    claims_corroborated: int = 0
    claims_unchanged: int = 0
    misaligned_count: int = 0
    subject_entity_ids: tuple[uuid.UUID, ...] = ()
    skipped_urls: tuple[str, ...] = ()
    """Sources that could not be fetched, so their records were not stored."""

    @property
    def claims_recorded(self) -> int:
        return self.claims_created + self.claims_corroborated
