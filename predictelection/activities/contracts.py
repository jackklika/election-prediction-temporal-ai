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

from predictelection.research.debates import ScrapedDebate
from predictelection.sql import EntityKind, ResearchRunStatus, SourceKind


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


class IngestDebateInput(Contract):
    debate: ScrapedDebate
    source_snapshot_id: uuid.UUID
    research_run_id: uuid.UUID | None = None
    asserted_by: str | None = None


class IngestDebateOutput(Contract):
    event_id: uuid.UUID
    event_created: bool = False
    """False means this debate was already known under this exact title."""

    participant_ids: tuple[uuid.UUID, ...] = ()
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
        default=None, description="Narrow to one kind, e.g. event or person."
    )
    occurred_after: datetime | None = Field(
        default=None, description="For events: earliest date to consider."
    )
    occurred_before: datetime | None = Field(
        default=None, description="For events: latest date to consider."
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


# --------------------------------------------------------------------------
# Workflow
# --------------------------------------------------------------------------


class ResearchDebatesInput(Contract):
    subject: str = Field(
        min_length=1,
        description="Politician or race to research, e.g. 'Abdul El-Sayed'.",
    )
    asserted_by: str | None = None


class ResearchDebatesOutput(Contract):
    research_run_id: uuid.UUID
    debates_found: int = 0
    debates_new: int = 0
    """Debates not already in the graph. The number that actually matters."""

    debates_already_known: int = 0
    """Re-discoveries. A run of all re-discoveries has learned nothing."""

    claims_created: int = 0
    claims_corroborated: int = 0
    claims_unchanged: int = 0
    misaligned_count: int = 0
    event_ids: tuple[uuid.UUID, ...] = ()
    skipped_urls: tuple[str, ...] = ()
    """Sources that could not be fetched, so their debates were not stored."""

    @property
    def claims_recorded(self) -> int:
        return self.claims_created + self.claims_corroborated
