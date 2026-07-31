"""What every ingestor shares: the write context, and the shape of a result.

A domain module's job is to say *which claims a record supports*. Everything
around that — binding each claim to the snapshot it came from, to the run that
found it and to whoever asserted it — is identical for debates, endorsements,
poll releases and CSV rows alike, and was previously a closure retyped inside
each ingest function.

Retyping it is how a domain quietly loses a citation: the closure is where
`source_snapshot_id` and `research_run_id` get attached, so a new domain that
writes its own gets to forget one. `IngestContext` makes that impossible to
express — there is no way to record a claim through it without the evidence.

`Ingestion` is deliberately not named after any domain. It answers "what did
this record put in the graph", which is the same question regardless of what the
record was, and it is what lets one activity report on every ingestor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid

from sqlalchemy.orm import Session

from predictelection.research.archive import SourceArchive
from predictelection.research.scraped import ScrapedEntity
from predictelection.sql import (
    ClaimAssertion,
    ClaimOutcome,
    EntityKind,
    EntityMention,
    EvidenceLocator,
    FullSourceLocator,
    PredicateValue,
    RecordedClaim,
    RecordOrigin,
    Resolution,
    SourceKind,
    SourceSnapshot,
    Validity,
    get_predicate_spec,
    record_claim_from_source,
    resolve_entity_mention,
)


@dataclass(frozen=True, slots=True)
class Ingestion:
    """What one scraped record put in the graph.

    One claim per assertion the source supports, never one per record: review
    has to be able to reject a wrong participant without discarding the event it
    was attached to, and a caller has to be able to count what was new.
    """

    subject_entity_id: uuid.UUID
    """The thing the record is chiefly about — the debate, the candidacy."""

    recorded: tuple[RecordedClaim, ...]
    subject_created: bool = False
    """False means the subject was already in the graph under this identity."""

    related_entity_ids: tuple[uuid.UUID, ...] = ()
    """Everything else the record resolved: participants, parties, offices."""

    registered_source_ids: tuple[uuid.UUID, ...] = ()
    """Sources noted for later fetching, such as a video the record linked."""

    @property
    def assertions(self) -> tuple[ClaimAssertion, ...]:
        return tuple(item.assertion for item in self.recorded)

    @property
    def misaligned(self) -> tuple[ClaimAssertion, ...]:
        """Stored but queued for review: the entity kinds did not fit the predicate.

        A quality signal about the extraction, not a failure count. Dropping
        these instead would hide exactly the cases worth looking at.
        """

        return tuple(a for a in self.assertions if not a.ontology_aligned)

    def count(self, outcome: ClaimOutcome) -> int:
        return sum(1 for item in self.recorded if item.outcome is outcome)


@dataclass(frozen=True, slots=True)
class IngestContext:
    """Everything a claim needs besides the claim itself.

    Constructed once by the caller that knows the provenance — an activity, an
    importer — and handed to a domain ingestor, which then cannot record
    anything unattributed.
    """

    session: Session
    snapshot: SourceSnapshot
    """The archived bytes every claim from this record will cite."""

    archive: SourceArchive | None = None
    """Needed only to register further sources a record points at."""

    research_run_id: uuid.UUID | None = None
    asserted_by: str | None = None
    origin: RecordOrigin = RecordOrigin.MODEL
    locator: EvidenceLocator = field(default_factory=FullSourceLocator)
    """Where in the snapshot, when a claim does not say something narrower.

    A whole-page default is honest for an agent that read the page as a whole.
    An importer should pass a per-row `JsonEvidenceLocator` to `record` instead,
    so a bad row can be traced to the field it came from rather than to the file.
    """

    def resolve(
        self, kind: EntityKind, named: ScrapedEntity | EntityMention | str
    ) -> Resolution:
        """Name to stable entity ID, carrying every identifier the source gave.

        Three inputs because there are three kinds of caller. A `ScrapedEntity`
        is what a source reported. A bare string is for things a record names
        without describing. An `EntityMention` is for identifiers the ingestor
        *derived* rather than read — office and election keys, which no source
        states and no model should be asked to guess.
        """

        if isinstance(named, EntityMention):
            mention = named
        elif isinstance(named, ScrapedEntity):
            mention = named.as_mention(kind)
        else:
            mention = EntityMention(kind=kind, name=named)
        return resolve_entity_mention(self.session, mention)

    def record(
        self,
        slug: str,
        *,
        subject_id: uuid.UUID,
        object_id: uuid.UUID | None = None,
        value: PredicateValue | dict[str, object] | None = None,
        validity: Validity | None = None,
        locator: EvidenceLocator | None = None,
        excerpt: str | None = None,
    ) -> RecordedClaim:
        """One attributed claim, deduplicated against the graph.

        `locator` and `excerpt` are per-claim rather than per-record: passing one
        locator for everything collapses every assertion onto a single evidence
        anchor, which satisfies "cites a snapshot" while losing the part that
        makes it checkable — where inside it.
        """

        return record_claim_from_source(
            self.session,
            predicate=get_predicate_spec(slug),
            subject_id=subject_id,
            object_id=object_id,
            value=value,
            validity=validity,
            source_snapshot_id=self.snapshot.id,
            locator=locator or self.locator,
            excerpt=excerpt,
            research_run_id=self.research_run_id,
            origin=self.origin,
            asserted_by=self.asserted_by,
        )

    def register_source(
        self, *, kind: SourceKind, canonical_url: str, title: str | None = None
    ) -> uuid.UUID | None:
        """Note a source without fetching it, for a later activity to pick up.

        Returns None when no archive was supplied, so an ingestor can offer the
        link unconditionally and a caller that only wants claims can decline.
        """

        if self.archive is None:
            return None
        return self.archive.source(
            kind=kind, canonical_url=canonical_url, title=title
        ).id
