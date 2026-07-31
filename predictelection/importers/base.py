"""The shape every importer takes, and the runner that drives it.

An importer is the deterministic half of ingestion: it reads a published file —
FEC bulk data, OCD divisions, certified results, poll releases — and turns rows
into claims without an LLM anywhere near them. Vote counts and poll percentages
should never come from a model when a CSV exists.

The provenance chain is identical to a scrape's, deliberately. The file is
archived as an `Artifact` exactly like a web page, so a claim from row 4,812 of
a CSV cites bytes that can be re-read and a locator inside them, the same as a
claim from a paragraph of an article. "It's just a CSV" is not a reason to skip
archiving; it is the reason a count can be checked years later.

The one thing that differs is the locator. An agent read a page as a whole and
cites `FullSourceLocator`; an importer knows exactly which row it was looking
at, so every claim cites a `JsonEvidenceLocator` pointing at that row. That is
what turns "this number came from the FEC" into "this number came from this
line, which you can go and read".

Importers run under a `ResearchRun` like any scrape, so
`ontology_alignment_score` stays answerable per import and a bad parse shows up
as a review queue rather than as silently wrong claims.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
from typing import ClassVar
import uuid

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from predictelection.research.archive import SourceArchive
from predictelection.research.ingestion import Ingestion, IngestContext
from predictelection.sql import (
    ClaimOutcome,
    JsonEvidenceLocator,
    RecordedClaim,
    RecordOrigin,
    ResearchRun,
    ResearchRunInput,
    ResearchRunStatus,
    SourceKind,
    SourceSnapshot,
    get_or_create,
    idempotency_key,
    ontology_alignment_score,
)
from predictelection.storage.base import ObjectStore, content_sha256


logger = logging.getLogger(__name__)

FETCH_TIMEOUT_SECONDS = 120.0
"""Longer than the archiver's: bulk files are tens of megabytes."""

IMPORT_INPUT_ROLE = "imported_file"
"""Names the run-to-snapshot link, so a retry can find the bytes it already read."""


@dataclass(frozen=True, slots=True)
class ImportRow:
    """One parsed row, and where it sits in the file that was archived."""

    index: int
    """Zero-based position among the rows this importer parsed."""

    data: Mapping[str, str]

    @property
    def locator(self) -> JsonEvidenceLocator:
        """A pointer at this row, so a wrong claim traces to the line it came from.

        A JSON pointer even for CSV: the archived artifact is the file, and the
        row index is the only stable coordinate inside it. The alternative — one
        `FullSourceLocator` for the whole import — collapses every claim from
        100,000 rows onto a single evidence anchor, which cites the file and
        tells you nothing about where in it to look.
        """

        return JsonEvidenceLocator(json_pointer=f"/rows/{self.index}")


@dataclass(frozen=True, slots=True)
class ImportResult:
    """What one import run did, in the same terms an ingestion reports."""

    research_run_id: uuid.UUID
    source_snapshot_id: uuid.UUID
    rows_read: int = 0
    rows_skipped: int = 0
    """Rows the importer's own filter excluded. Logged, never silent."""

    rows_failed: int = 0
    """Rows that raised. Recorded rather than aborting the whole file."""

    entities_touched: tuple[uuid.UUID, ...] = ()
    recorded: tuple[RecordedClaim, ...] = field(default=())
    alignment: float | None = None

    def count(self, outcome: ClaimOutcome) -> int:
        return sum(1 for item in self.recorded if item.outcome is outcome)

    @property
    def claims_created(self) -> int:
        return self.count(ClaimOutcome.CREATED)

    @property
    def misaligned_count(self) -> int:
        return sum(1 for item in self.recorded if not item.assertion.ontology_aligned)


class Importer(ABC):
    """Fetch a published file, parse it, turn rows into claims.

    Subclasses supply `source_url`, `parse` and `ingest`. Everything else —
    archiving, the research run, per-row evidence, counting, and not letting one
    bad row abort 100,000 good ones — comes from `run_import`.
    """

    name: ClassVar[str]
    """Names the ResearchRun, so an import is scoreable like any other run."""

    source_kind: ClassVar[SourceKind] = SourceKind.DATASET
    media_type: ClassVar[str | None] = None

    @property
    @abstractmethod
    def source_url(self) -> str:
        """The published file. Archived, and cited by every claim below it."""

    @property
    def subject(self) -> str:
        """What this run covers, for the ResearchRun row. Usually the URL."""

        return self.source_url

    def fetch(self, http: httpx.Client) -> bytes:
        """Read the bytes. Overridable for anything not a plain GET."""

        response = http.get(self.source_url)
        response.raise_for_status()
        return response.content

    @abstractmethod
    def parse(self, raw: bytes) -> Iterator[ImportRow] | FilteredParse:
        """Rows worth importing, in file order.

        Filtering belongs here, and whatever it drops must be counted — return
        a `FilteredParse` rather than a bare iterator. A silent filter reads
        downstream as "the source did not contain that", which is
        indistinguishable from a bug.
        """

    @abstractmethod
    def ingest(self, row: ImportRow, context: IngestContext) -> Ingestion:
        """Resolve this row's entities and record its claims."""


@dataclass(frozen=True, slots=True)
class FilteredParse:
    """Rows kept, and how many were deliberately dropped.

    Returned by importers that filter, so `run_import` can log the number rather
    than let a 100,000-row file quietly become 3,000 rows.
    """

    rows: Sequence[ImportRow]
    skipped: int = 0


def run_import(
    session: Session,
    store: ObjectStore,
    importer: Importer,
    *,
    http: httpx.Client | None = None,
    raw: bytes | None = None,
    asserted_by: str | None = None,
    retrieved_at: datetime | None = None,
) -> ImportResult:
    """Archive the file, then turn every row into attributed claims.

    `raw` lets a caller supply bytes it already has — a test fixture, a file on
    disk, a download done elsewhere. The archiving and citation path is the same
    either way, so a fixture-driven test exercises the real provenance chain.

    Re-running is safe and is the property that matters: the artifact
    deduplicates on its SHA-256, entities resolve to the ones already there, and
    `record_claim_from_source` returns UNCHANGED for anything this run already
    asserted. A second run of an unchanged file writes nothing.
    """

    if raw is None:
        client = http or httpx.Client(
            timeout=FETCH_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={"User-Agent": "predictelection/0.1 (importer)"},
        )
        raw = importer.fetch(client)

    # Scoped to the file's content, which is what makes a re-run mean the right
    # thing. Re-importing an unchanged file is a retry and shares this run;
    # importing an updated file is new research and gets its own, so the two are
    # still tellable apart and per-run alignment stays answerable.
    key = idempotency_key(
        "import",
        importer=importer.name,
        subject=importer.subject,
        content=content_sha256(raw),
    )
    run, opened = get_or_create(
        session,
        ResearchRun(
            idempotency_key=key,
            task_type=importer.name,
            status=ResearchRunStatus.RUNNING,
            input_data={"subject": importer.subject},
        ),
        key=ResearchRun.idempotency_key == key,
    )

    snapshot = _snapshot_for(
        session,
        store,
        importer,
        raw,
        run=run,
        reuse=not opened,
        retrieved_at=retrieved_at,
    )

    parsed = importer.parse(raw)
    rows, skipped = (
        (list(parsed.rows), parsed.skipped)
        if isinstance(parsed, FilteredParse)
        else (list(parsed), 0)
    )
    if skipped:
        logger.info(
            "%s: kept %d rows, filtered %d — the rest of the file is not imported",
            importer.name,
            len(rows),
            skipped,
        )

    recorded: list[RecordedClaim] = []
    entities: list[uuid.UUID] = []
    failed = 0

    for row in rows:
        context = IngestContext(
            session=session,
            snapshot=snapshot,
            research_run_id=run.id,
            asserted_by=asserted_by or importer.name,
            # IMPORT, not MODEL: nothing here was inferred, and the distinction
            # is what lets review triage a parse bug differently from a
            # hallucination.
            origin=RecordOrigin.IMPORT,
            locator=row.locator,
        )
        try:
            ingestion = importer.ingest(row, context)
        except Exception:
            # One malformed row must not cost the other 99,999. It is logged
            # with its index, which is also the locator, so it can be read.
            failed += 1
            logger.exception("%s: row %d failed", importer.name, row.index)
            continue
        recorded.extend(ingestion.recorded)
        entities.append(ingestion.subject_entity_id)
        entities.extend(ingestion.related_entity_ids)

    session.flush()
    run.status = ResearchRunStatus.SUCCEEDED
    run.completed_at = datetime.now(UTC)

    return ImportResult(
        research_run_id=run.id,
        source_snapshot_id=snapshot.id,
        rows_read=len(rows),
        rows_skipped=skipped,
        rows_failed=failed,
        entities_touched=tuple(dict.fromkeys(entities)),
        recorded=tuple(recorded),
        alignment=ontology_alignment_score(session, research_run_id=run.id),
    )


def _snapshot_for(
    session: Session,
    store: ObjectStore,
    importer: Importer,
    raw: bytes,
    *,
    run: ResearchRun,
    reuse: bool,
    retrieved_at: datetime | None,
) -> SourceSnapshot:
    """The archived bytes this run cites, stable across retries of that run.

    A snapshot is an *observation*, keyed on (source, artifact, retrieved_at),
    so observing the same file twice legitimately makes two of them — "still
    unchanged today" is itself evidence. That is right for a scrape and wrong
    for a retry: the evidence anchor is fingerprinted over the snapshot, so a
    fresh one on every attempt gives every claim a new anchor, and a retried
    import writes a second assertion for every row instead of nothing.

    So a run that was already open reuses the snapshot it recorded as its input.
    The link is `ResearchRunInput`, which exists to say exactly this: the exact
    bytes this run consumed.
    """

    if reuse:
        existing = session.scalars(
            select(SourceSnapshot)
            .join(
                ResearchRunInput,
                ResearchRunInput.source_snapshot_id == SourceSnapshot.id,
            )
            .where(
                ResearchRunInput.research_run_id == run.id,
                ResearchRunInput.role == IMPORT_INPUT_ROLE,
            )
            .order_by(SourceSnapshot.retrieved_at)
            .limit(1)
        ).first()
        if existing is not None:
            return existing

    snapshot = SourceArchive(session, store).observe(
        kind=importer.source_kind,
        canonical_url=importer.source_url,
        content=raw,
        media_type=importer.media_type,
        retrieved_at=retrieved_at or datetime.now(UTC),
    )
    session.flush()
    get_or_create(
        session,
        ResearchRunInput(
            research_run_id=run.id,
            source_snapshot_id=snapshot.id,
            role=IMPORT_INPUT_ROLE,
        ),
        key=(ResearchRunInput.research_run_id == run.id)
        & (ResearchRunInput.source_snapshot_id == snapshot.id)
        & (ResearchRunInput.role == IMPORT_INPUT_ROLE),
    )
    return snapshot


def rows_from_delimited(
    raw: bytes,
    *,
    fieldnames: Sequence[str] | None = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> Iterator[ImportRow]:
    """CSV or pipe-delimited bytes to rows, index preserved.

    `fieldnames` is for headerless files — FEC bulk data ships without a header
    row, so the column names come from its documentation rather than the file.
    """

    import csv
    import io

    text = io.StringIO(raw.decode(encoding, errors="replace"), newline="")
    reader = csv.DictReader(
        text,
        fieldnames=list(fieldnames) if fieldnames else None,
        delimiter=delimiter,
    )
    for index, data in enumerate(reader):
        yield ImportRow(
            index=index,
            data={k: (v or "").strip() for k, v in data.items() if k is not None},
        )


__all__ = [
    "FETCH_TIMEOUT_SECONDS",
    "FilteredParse",
    "ImportResult",
    "ImportRow",
    "Importer",
    "rows_from_delimited",
    "run_import",
]
