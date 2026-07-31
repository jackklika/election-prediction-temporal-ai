"""Debates: the scraped shape, and how it becomes claims.

This is the template for every other domain. The pattern is worth copying:

- The Pydantic models describe what an agent *observed*, in the agent's terms.
  They carry names, not IDs, because a scraper has no way to know an ID.
- Ingestion resolves each name to an entity, then emits one claim per assertion
  the source actually supports. Nothing is inferred here — a debate's existence
  and its participants are separate claims because a source can get one right
  and the other wrong, and review has to be able to reject them separately.
- Every claim carries the snapshot it came from, so all of it is attributable.

Precision is explicit rather than assumed: an agent that only read "September
2026" must say so, otherwise the graph silently gains a false midnight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field

from predictelection.sql import (
    ClaimAssertion,
    EntityKind,
    EntityMention,
    EvidenceLocator,
    EventOccurrenceStatus,
    ExternalIdentifier,
    FullSourceLocator,
    PoliticalEventKind,
    RecordOrigin,
    SourceKind,
    SourceSnapshot,
    TimePrecision,
    Validity,
    get_predicate_spec,
    record_claim_from_source,
    resolve_entity_mention,
)
from predictelection.research.archive import SourceArchive


class ScrapedEntity(BaseModel):
    """A named thing an agent saw, with an external ID when the source gave one."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=500)
    wikidata_id: str | None = Field(default=None, pattern=r"^Q[0-9]+$")

    def as_mention(self, kind: EntityKind) -> EntityMention:
        identifiers = (
            (ExternalIdentifier(namespace="wikidata", value=self.wikidata_id),)
            if self.wikidata_id
            else ()
        )
        return EntityMention(kind=kind, name=self.name, identifiers=identifiers)


class ScrapedDebate(BaseModel):
    """One debate as an agent reported it."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    starts_at: datetime
    starts_at_precision: TimePrecision = TimePrecision.DAY
    ends_at: datetime | None = None
    status: EventOccurrenceStatus = EventOccurrenceStatus.OCCURRED
    participants: tuple[ScrapedEntity, ...] = ()
    contest: ScrapedEntity | None = None
    jurisdiction: ScrapedEntity | None = None
    video_url: str | None = None
    """Recorded as a source to fetch later, not archived here."""


@dataclass(frozen=True, slots=True)
class DebateIngestion:
    event_id: uuid.UUID
    assertions: tuple[ClaimAssertion, ...]
    participant_ids: tuple[uuid.UUID, ...] = field(default=())
    video_source_id: uuid.UUID | None = None

    @property
    def misaligned(self) -> tuple[ClaimAssertion, ...]:
        return tuple(a for a in self.assertions if not a.ontology_aligned)


def ingest_debate(
    session,
    *,
    debate: ScrapedDebate,
    snapshot: SourceSnapshot,
    archive: SourceArchive | None = None,
    research_run_id: uuid.UUID | None = None,
    asserted_by: str | None = None,
    locator: EvidenceLocator | None = None,
    origin: RecordOrigin = RecordOrigin.MODEL,
) -> DebateIngestion:
    """Turn one scraped debate into resolved entities and attributed claims."""

    where = locator or FullSourceLocator()

    def assert_claim(
        slug: str, *, subject_id, object_id=None, value=None, validity=None
    ):
        return record_claim_from_source(
            session,
            predicate=get_predicate_spec(slug),
            subject_id=subject_id,
            object_id=object_id,
            value=value,
            validity=validity,
            source_snapshot_id=snapshot.id,
            locator=where,
            excerpt=debate.title,
            research_run_id=research_run_id,
            origin=origin,
            asserted_by=asserted_by,
        )

    event = resolve_entity_mention(
        session,
        EntityMention(kind=EntityKind.EVENT, name=debate.title),
    )
    assertions = [
        assert_claim(
            "event_kind",
            subject_id=event.entity_id,
            value={"kind": PoliticalEventKind.DEBATE},
        ),
        assert_claim(
            "event_occurrence",
            subject_id=event.entity_id,
            value={"status": debate.status},
            validity=Validity.between(
                debate.starts_at, debate.ends_at, debate.starts_at_precision
            ),
        ),
    ]

    participant_ids: list[uuid.UUID] = []
    for person in debate.participants:
        resolved = resolve_entity_mention(session, person.as_mention(EntityKind.PERSON))
        participant_ids.append(resolved.entity_id)
        assertions.append(
            assert_claim(
                "participated_in",
                subject_id=resolved.entity_id,
                object_id=event.entity_id,
            )
        )

    if debate.contest is not None:
        contest = resolve_entity_mention(
            session, debate.contest.as_mention(EntityKind.CONTEST)
        )
        assertions.append(
            assert_claim(
                "event_about_contest",
                subject_id=event.entity_id,
                object_id=contest.entity_id,
            )
        )

    if debate.jurisdiction is not None:
        jurisdiction = resolve_entity_mention(
            session, debate.jurisdiction.as_mention(EntityKind.JURISDICTION)
        )
        assertions.append(
            assert_claim(
                "event_in_jurisdiction",
                subject_id=event.entity_id,
                object_id=jurisdiction.entity_id,
            )
        )

    video_source_id = None
    if debate.video_url is not None and archive is not None:
        # Registered, not fetched: downloading and transcribing is a separate
        # activity, and this is the row it will look for.
        video_source_id = archive.source(
            kind=SourceKind.VIDEO,
            canonical_url=debate.video_url,
            title=debate.title,
        ).id

    return DebateIngestion(
        event_id=event.entity_id,
        assertions=tuple(assertions),
        participant_ids=tuple(participant_ids),
        video_source_id=video_source_id,
    )
