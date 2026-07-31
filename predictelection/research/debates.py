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

from datetime import datetime
from typing import Literal
import uuid

from pydantic import Field

from predictelection.sql import (
    EntityKind,
    EventOccurrenceStatus,
    ParticipationRole,
    PoliticalEventKind,
    SourceKind,
    TimePrecision,
    Validity,
)
from predictelection.research.ingestion import Ingestion, IngestContext
from predictelection.research.scraped import ScrapedEntity, ScrapedRecord


class ScrapedDebate(ScrapedRecord):
    """One debate as an agent reported it.

    Doubles as the agent's output contract, so the field descriptions are the
    prompt: there is no mapping layer between what the model emits and what gets
    stored, and no second place for the two to drift apart.
    """

    record_type: Literal["debate"] = "debate"

    title: str = Field(
        min_length=1, max_length=500, description="Formal name or title of the debate."
    )
    starts_at: datetime = Field(description="When the debate began, with a timezone.")
    starts_at_precision: TimePrecision = Field(
        default=TimePrecision.DAY,
        description=(
            "How precisely the source gave the time. Use 'day' when only a date "
            "was stated and 'minute' when a clock time was. Do not claim more "
            "precision than the source did."
        ),
    )
    ends_at: datetime | None = Field(
        default=None, description="When it ended, if the source says."
    )
    status: EventOccurrenceStatus = Field(
        default=EventOccurrenceStatus.OCCURRED,
        description="Whether it happened, is upcoming, was postponed, or cancelled.",
    )
    participants: tuple[ScrapedEntity, ...] = Field(
        default=(), description="The candidates who debated each other."
    )
    moderators: tuple[ScrapedEntity, ...] = Field(
        default=(),
        description=(
            "Anyone who ran the debate rather than competing in it — moderators, "
            "panelists, hosts. Do not list them as participants."
        ),
    )
    contest: ScrapedEntity | None = Field(
        default=None,
        description="The race being contested, e.g. 'Michigan Governor 2026'.",
    )
    jurisdiction: ScrapedEntity | None = Field(
        default=None, description="Where it was held, e.g. 'Michigan'."
    )
    video_url: str | None = Field(
        default=None,
        description="Full recording, ideally the host's official upload.",
    )

    @property
    def source_title(self) -> str | None:
        return self.title


def ingest_debate(debate: ScrapedDebate, context: IngestContext) -> Ingestion:
    """Turn one scraped debate into resolved entities and attributed claims.

    Two positional arguments, matching every other ingestor, so the registry can
    dispatch to any of them without knowing which domain it is holding.
    """

    event = context.resolve(EntityKind.EVENT, debate.title)
    recorded = [
        context.record(
            "event_kind",
            subject_id=event.entity_id,
            value={"kind": PoliticalEventKind.DEBATE},
            excerpt=debate.title,
        ),
        context.record(
            "event_occurrence",
            subject_id=event.entity_id,
            value={"status": debate.status},
            validity=Validity.between(
                debate.starts_at, debate.ends_at, debate.starts_at_precision
            ),
            excerpt=debate.title,
        ),
    ]

    # Role is what separates a debater from the person asking the questions.
    participant_ids: list[uuid.UUID] = []
    for people, role in (
        (debate.participants, ParticipationRole.CANDIDATE),
        (debate.moderators, ParticipationRole.MODERATOR),
    ):
        for person in people:
            resolved = context.resolve(EntityKind.PERSON, person)
            participant_ids.append(resolved.entity_id)
            recorded.append(
                context.record(
                    "participated_in",
                    subject_id=resolved.entity_id,
                    object_id=event.entity_id,
                    value={"role": role},
                    excerpt=person.name,
                )
            )

    if debate.contest is not None:
        contest = context.resolve(EntityKind.CONTEST, debate.contest)
        recorded.append(
            context.record(
                "event_about_contest",
                subject_id=event.entity_id,
                object_id=contest.entity_id,
                excerpt=debate.contest.name,
            )
        )

    if debate.jurisdiction is not None:
        jurisdiction = context.resolve(EntityKind.JURISDICTION, debate.jurisdiction)
        recorded.append(
            context.record(
                "event_in_jurisdiction",
                subject_id=event.entity_id,
                object_id=jurisdiction.entity_id,
                excerpt=debate.jurisdiction.name,
            )
        )

    video_source_id = None
    if debate.video_url is not None:
        # Registered, not fetched: downloading and transcribing is a separate
        # activity, and this is the row it will look for.
        video_source_id = context.register_source(
            kind=SourceKind.VIDEO,
            canonical_url=debate.video_url,
            title=debate.title,
        )

    return Ingestion(
        subject_entity_id=event.entity_id,
        recorded=tuple(recorded),
        subject_created=event.created,
        related_entity_ids=tuple(participant_ids),
        registered_source_ids=() if video_source_id is None else (video_source_id,),
    )
