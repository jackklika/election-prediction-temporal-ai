"""Race structure: what a contest *is*, so contests stop floating.

Before this, a debate pointed at a contest with no office, no stage, no party
and no candidates. Nothing could be correlated across races, which is the whole
point of building a graph rather than a list.

The record asks for the *components* of a contest rather than for a contest key,
and the ingestor derives every identifier itself. That is deliberate. A model
handed a key format will produce plausible keys, and a plausible-but-wrong key
mints a contest that looks identified and that nothing else will ever resolve
to — worse than no key at all. Division, office, cycle and stage are things a
source states; the key is arithmetic on them, and arithmetic belongs in code.

Deriving rather than asking also gets `advances_to` for free: a primary knows
its own general, because they differ only in the stage segment.
"""

from __future__ import annotations

from typing import Literal
import uuid

from pydantic import Field

from predictelection.research.contests import (
    CONTEST_KEY_NAMESPACE,
    ELECTION_KEY_NAMESPACE,
    OFFICE_KEY_NAMESPACE,
    ContestKey,
    ElectionKey,
    OfficeKey,
)
from predictelection.research.ingestion import Ingestion, IngestContext
from predictelection.research.scraped import ScrapedEntity, ScrapedRecord
from predictelection.sql import (
    ContestStage,
    EntityKind,
    EntityMention,
    ExternalIdentifier,
)


class ScrapedRaceStructure(ScrapedRecord):
    """One contest, described by what it is rather than by what it is called."""

    record_type: Literal["race_structure"] = "race_structure"

    division_id: str = Field(
        pattern=r"^ocd-division/country:[a-z]{2}(?:/[a-z_]+:[^/]+)*$",
        description=(
            "Open Civic Data ID of where the race is held, e.g. "
            "'ocd-division/country:us/state:mi'. For a US House race this is "
            "the district — '.../state:mi/cd:11' — not the state."
        ),
    )
    office: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "The seat itself, without the place or the year: 'Governor', "
            "'US Senate', 'Attorney General'. Not 'Governor of Michigan'."
        ),
    )
    cycle: int = Field(
        ge=1788,
        le=2200,
        description="The year the election is held, e.g. 2026.",
    )
    stage: ContestStage = Field(
        description=(
            "Which round this is. A primary and a general are different "
            "contests with different candidates and different outcomes, so "
            "never describe both as one."
        )
    )
    party: str | None = Field(
        default=None,
        max_length=100,
        description=(
            "Whose primary, e.g. 'Democratic'. Must be null for a general "
            "election, which is contested between parties rather than within "
            "one."
        ),
    )
    jurisdiction_name: str = Field(
        min_length=1,
        max_length=500,
        description="What the source calls the place, e.g. 'Michigan'.",
    )
    advances_to: ContestStage | None = Field(
        default=None,
        description=(
            "The round this one's winner goes on to, usually 'general' for a "
            "primary. Leave null if this is the final round or the source does "
            "not say."
        ),
    )

    @property
    def source_title(self) -> str | None:
        return f"{self.office} {self.cycle}"


def ingest_race_structure(
    record: ScrapedRaceStructure, context: IngestContext
) -> Ingestion:
    """Turn one described race into a contest joined to everything around it."""

    key = ContestKey.build(
        division=record.division_id,
        office=record.office,
        cycle=record.cycle,
        stage=record.stage,
        party=record.party,
    )
    contest = context.resolve(EntityKind.CONTEST, _contest_mention(key))

    jurisdiction = context.resolve(
        EntityKind.JURISDICTION,
        # ocd_id, so this lands on the entity the OCD import already created
        # rather than minting a second Michigan under whatever the source
        # happened to call it.
        ScrapedEntity(name=record.jurisdiction_name, ocd_id=key.division),
    )
    office = context.resolve(EntityKind.OFFICE, _office_mention(key.office_key))
    election = context.resolve(EntityKind.ELECTION, _election_mention(key.election_key))

    related = [jurisdiction.entity_id, office.entity_id, election.entity_id]
    recorded = [
        context.record(
            "contest_stage",
            subject_id=contest.entity_id,
            value={"stage": key.stage},
            excerpt=record.office,
        ),
        context.record(
            "contest_in_jurisdiction",
            subject_id=contest.entity_id,
            object_id=jurisdiction.entity_id,
            excerpt=record.jurisdiction_name,
        ),
        context.record(
            "contest_for_office",
            subject_id=contest.entity_id,
            object_id=office.entity_id,
            excerpt=record.office,
        ),
        context.record(
            "contest_of_election",
            subject_id=contest.entity_id,
            object_id=election.entity_id,
            excerpt=record.office,
        ),
    ]

    if key.party is not None:
        party = context.resolve(EntityKind.PARTY, key.party.title())
        related.append(party.entity_id)
        recorded.append(
            context.record(
                "contest_party",
                subject_id=contest.entity_id,
                object_id=party.entity_id,
                excerpt=record.party,
            )
        )

    if record.advances_to is not None and record.advances_to is not key.stage:
        # Derived, not asked for: the successor differs from this contest only
        # in its stage, so naming it would be a chance to name it differently.
        successor_key = key.at_stage(record.advances_to)
        successor = context.resolve(EntityKind.CONTEST, _contest_mention(successor_key))
        related.append(successor.entity_id)
        recorded.append(
            context.record(
                "advances_to",
                subject_id=contest.entity_id,
                object_id=successor.entity_id,
                excerpt=record.office,
            )
        )

    return Ingestion(
        subject_entity_id=contest.entity_id,
        recorded=tuple(recorded),
        subject_created=contest.created,
        related_entity_ids=tuple(dict.fromkeys(related)),
    )


def _contest_mention(key: ContestKey) -> EntityMention:
    return _derived(EntityKind.CONTEST, CONTEST_KEY_NAMESPACE, str(key), key.label)


def _office_mention(key: OfficeKey) -> EntityMention:
    return _derived(EntityKind.OFFICE, OFFICE_KEY_NAMESPACE, str(key), key.label)


def _election_mention(key: ElectionKey) -> EntityMention:
    return _derived(EntityKind.ELECTION, ELECTION_KEY_NAMESPACE, str(key), key.label)


def _derived(kind: EntityKind, namespace: str, value: str, label: str) -> EntityMention:
    """A mention identified by a key we computed rather than a name we read.

    The label is only what a newly minted entity gets called. Identity is the
    identifier, so a source that phrases it differently still resolves here.
    """

    return EntityMention(
        kind=kind,
        name=label,
        identifiers=(ExternalIdentifier(namespace=namespace, value=value),),
        # Same argument as for events: a derived key is the definition of the
        # entity, so a key we have not seen is a new entity even if its label
        # happens to match one.
        identifiers_are_authoritative=True,
    )


__all__ = ["ScrapedRaceStructure", "ingest_race_structure"]


def contest_id_for(context: IngestContext, key: ContestKey) -> uuid.UUID:
    """Resolve a contest by key alone, for callers holding one already."""

    return context.resolve(EntityKind.CONTEST, _contest_mention(key)).entity_id
