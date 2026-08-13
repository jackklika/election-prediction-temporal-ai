"""Endorsements: who backed whom, and for how long.

The first writer for `endorsed`, and the roadmap's Phase 4 requirement is the
test case that motivated it: **an endorsement and its later withdrawal are two
claims, both retrievable, with distinct validity.** Never an update, never a
delete — the graph has to keep answering "who had this endorsement in June"
after the endorsement is gone.

The 2026 Wisconsin Democratic gubernatorial primary is that shape twice over.
David Crowley withdrew and endorsed Sara Rodriguez, then re-entered and beat
her: his endorsement is true over exactly the window he was out of the race.
Missy Hughes endorsed Rodriguez and later Crowley — two live endorsements of
opposing candidates, correct at different times, and contradictory only if the
intervals are thrown away.

A withdrawal is expressed as a second claim carrying
`EndorsementStrength.WITHDRAWN` over the later interval, which is what that enum
member exists for. Its own record type rather than a field on the candidacy: an
endorsement is a fact about the endorser, it outlives any one race, and a future
endorsements agent reading press releases should reuse this shape rather than
invent a second one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from predictelection.research.contests import ContestKey
from predictelection.research.ingestion import Ingestion, IngestContext
from predictelection.research.scraped import (
    ScrapedDateTime,
    ScrapedEntity,
    ScrapedRecord,
)
from predictelection.sql import (
    ContestStage,
    EndorsementStrength,
    EntityKind,
    TimePrecision,
    Validity,
)


class ScrapedEndorsement(ScrapedRecord):
    """One endorsement, over the period the source says it held."""

    record_type: Literal["endorsement"] = "endorsement"

    endorser: ScrapedEntity = Field(
        description=(
            "Who gave the endorsement — a person, an organization or a party, "
            "named exactly as the source writes them."
        )
    )
    endorser_kind: EntityKind = Field(
        default=EntityKind.PERSON,
        description=(
            "Whether the endorser is a person, organization or party. "
            "Newspapers and unions are organizations, not people."
        ),
    )
    endorsee: ScrapedEntity = Field(description="The candidate being backed.")
    strength: EndorsementStrength = Field(
        default=EndorsementStrength.FULL,
        description=(
            "'full' for a plain endorsement, 'qualified' when the endorser "
            "stated reservations, 'withdrawn' when this claim records the "
            "endorsement being taken back. A withdrawal is a SEPARATE record "
            "over the later period, never an edit to the original."
        ),
    )
    context: str | None = Field(
        default=None,
        max_length=500,
        description="How the source characterized it, if it adds anything.",
    )

    # The race, so the endorsement is anchored to a contest rather than floating.
    division_id: str = Field(
        pattern=r"^ocd-division/country:[a-z]{2}(?:/[a-z_]+:[^/]+)*$",
        description="Open Civic Data ID of where the race is held.",
    )
    office: str = Field(min_length=1, max_length=100)
    cycle: int = Field(ge=1788, le=2200)
    stage: ContestStage
    party: str | None = Field(default=None, max_length=100)

    announced_on: ScrapedDateTime | None = Field(
        default=None,
        description=(
            "When the endorsement was made. Null when the source does not date "
            "it — never guess, an invented date is a false fact about timing."
        ),
    )
    announced_precision: TimePrecision = Field(
        default=TimePrecision.DAY,
        description=(
            "How precisely the source gave the date. 'month' when it says only "
            "'mid-July'; do not promote a vague date to a day."
        ),
    )
    ended_on: ScrapedDateTime | None = Field(
        default=None,
        description=(
            "When it stopped holding — the day it was withdrawn, or the "
            "election. Null when it ran to the end or the source does not say."
        ),
    )
    ended_precision: TimePrecision = Field(default=TimePrecision.DAY)

    def validity(self) -> Validity:
        if self.announced_on is None and self.ended_on is None:
            return Validity.timeless()
        if self.announced_on is None:
            return Validity(end=self.ended_on, end_precision=self.ended_precision)
        return Validity.between(
            self.announced_on,
            self.ended_on,
            self.announced_precision,
            end_precision=self.ended_precision if self.ended_on else None,
        )

    def contest_key(self) -> ContestKey:
        return ContestKey.build(
            division=self.division_id,
            office=self.office,
            cycle=self.cycle,
            stage=self.stage,
            party=self.party,
        )


def ingest_endorsement(record: ScrapedEndorsement, context: IngestContext) -> Ingestion:
    """One endorsement claim, and the contest it belongs to.

    `endorsed` is a claim about two people, so the contest is not part of the
    proposition — it is recorded as a separate `candidate_in`-adjacent fact by
    whoever owns candidacies. What the contest gives us here is the *entity* the
    endorsee resolves against, so an endorsement and a candidacy in the same
    race meet on one person.
    """

    key = record.contest_key()
    endorser = context.resolve(record.endorser_kind, record.endorser)
    endorsee = context.resolve(EntityKind.PERSON, record.endorsee)
    contest = context.resolve(
        EntityKind.CONTEST,
        ScrapedEntity(name=key.label, contest_key=str(key)),
    )

    value: dict[str, object] = {"strength": record.strength}
    if record.context:
        value["context"] = record.context

    recorded = context.record(
        "endorsed",
        subject_id=endorser.entity_id,
        object_id=endorsee.entity_id,
        value=value,
        validity=record.validity(),
        excerpt=f"{record.endorser.name} → {record.endorsee.name}",
    )
    return Ingestion(
        subject_entity_id=endorser.entity_id,
        recorded=(recorded,),
        subject_created=endorser.created,
        related_entity_ids=(endorsee.entity_id, contest.entity_id),
    )
