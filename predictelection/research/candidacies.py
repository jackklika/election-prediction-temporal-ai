"""Candidacies: who ran, when they were in the race, and how it ended.

The 2026 Wisconsin Democratic gubernatorial primary is the shape this exists
for: David Crowley announced, withdrew and endorsed Rodriguez, re-entered, and
won; Rodriguez withdrew after his return but stayed on the ballot and took 4%.
None of that is expressible as a single boolean — it is a *timeline*, and the
claim model already speaks time:

- Each stint in the race is its own `candidate_in` claim over its own validity
  interval. In → out → in is two claims, both true forever over their windows.
  A withdrawal is never a deletion; the graph must still answer "who was
  running in June".
- "Won" is a `contest_result` claim with value `{won: true}` — asserted only
  when the source states an outcome, never inferred from vote totals, which is
  the roadmap's rule (multi-winner contests exist) and also this race's lesson:
  Rodriguez *withdrew* and still out-polled candidates who stayed in.
- Vote counts are deliberately absent here. They come from the results
  *importer*, deterministically; an agent asserting numbers is the one thing
  the pipeline forbids. The two writers' claims meet on the same contest and
  person entities via derived keys.

Dates carry their stated precision, and an unknown date is an open interval
end, not an invented day.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from predictelection.research.contests import ContestKey
from predictelection.research.ingestion import Ingestion, IngestContext
from predictelection.research.scraped import ScrapedEntity, ScrapedModel, ScrapedRecord
from predictelection.sql import (
    ContestStage,
    EntityKind,
    TimePrecision,
    Validity,
)


class CandidacyOutcome(StrEnum):
    """How the candidacy ended, as the source states it."""

    NOMINATED = "nominated"
    """Won this contest. The only outcome that asserts a contest_result."""

    ELIMINATED = "eliminated"
    """Stood through the vote and lost."""

    WITHDREW = "withdrew"
    """Left the race before it resolved. May still appear in results if the
    withdrawal came after ballots were set."""

    DISQUALIFIED = "disqualified"
    PENDING = "pending"
    """The contest has not resolved yet."""


class CandidacyStint(ScrapedModel):
    """One continuous period of being a candidate.

    Plural on the record because re-entry is real: a candidate who left and
    came back has two stints, and collapsing them into one interval would
    assert they were running while they were endorsing someone else.
    """

    entered_on: datetime | None = Field(
        default=None,
        description=(
            "When this stint began — announcement or re-entry. Null when the "
            "source does not date it."
        ),
    )
    entered_precision: TimePrecision = Field(
        default=TimePrecision.DAY,
        description="How precisely the source gave the entry date.",
    )
    left_on: datetime | None = Field(
        default=None,
        description=(
            "When this stint ended — withdrawal. Null when it ran to the "
            "election or is still running."
        ),
    )
    left_precision: TimePrecision = Field(
        default=TimePrecision.DAY,
        description="How precisely the source gave the exit date.",
    )

    def validity(self) -> Validity:
        if self.entered_on is None and self.left_on is None:
            return Validity.timeless()
        if self.entered_on is None:
            # An exit with an unknown start: say only what the source said.
            return Validity(end=self.left_on, end_precision=self.left_precision)
        return Validity.between(
            self.entered_on,
            self.left_on,
            self.entered_precision,
            end_precision=self.left_precision if self.left_on else None,
        )


class ScrapedCandidacy(ScrapedRecord):
    """One person's whole relationship with one contest."""

    record_type: Literal["candidacy"] = "candidacy"

    candidate: ScrapedEntity = Field(
        description="The person, named exactly as the source writes them."
    )
    division_id: str = Field(
        pattern=r"^ocd-division/country:[a-z]{2}(?:/[a-z_]+:[^/]+)*$",
        description=(
            "Open Civic Data ID of where the race is held, e.g. "
            "'ocd-division/country:us/state:wi'."
        ),
    )
    office: str = Field(
        min_length=1,
        max_length=100,
        description="The seat, without place or year: 'Governor', 'US Senate'.",
    )
    cycle: int = Field(ge=1788, le=2200)
    stage: ContestStage
    party: str | None = Field(
        default=None,
        max_length=100,
        description="Whose primary. Null for a general election.",
    )
    stints: tuple[CandidacyStint, ...] = Field(
        min_length=1,
        description=(
            "Every continuous period they were in the race, in order. A "
            "candidate who withdrew and re-entered has two. Dates only when "
            "the source states them."
        ),
    )
    outcome: CandidacyOutcome = Field(
        default=CandidacyOutcome.PENDING,
        description=(
            "How it ended, only as the source states it — 'nominee' means "
            "nominated; listed under 'Withdrawn' means withdrew."
        ),
    )
    remained_on_ballot: bool = Field(
        default=False,
        description=(
            "True when a withdrawal came too late to leave the ballot, so "
            "results still show votes for them."
        ),
    )

    @model_validator(mode="after")
    def _withdrew_means_an_exit(self):
        if self.outcome is CandidacyOutcome.WITHDREW and self.stints[-1].left_on is None:
            # Tolerated, not rejected: the source may state the withdrawal
            # without dating it. The interval stays open and the outcome still
            # records what happened.
            pass
        return self


def ingest_candidacy(record: ScrapedCandidacy, context: IngestContext) -> Ingestion:
    """One candidacy into interval claims, and its outcome when stated."""

    key = ContestKey.build(
        division=record.division_id,
        office=record.office,
        cycle=record.cycle,
        stage=record.stage,
        party=record.party,
    )
    person = context.resolve(EntityKind.PERSON, record.candidate)
    contest = context.resolve(
        EntityKind.CONTEST,
        ScrapedEntity(name=key.label, contest_key=str(key)),
    )

    recorded = [
        context.record(
            "candidate_in",
            subject_id=person.entity_id,
            object_id=contest.entity_id,
            validity=stint.validity(),
            excerpt=record.candidate.name,
        )
        for stint in record.stints
    ]

    if record.outcome is CandidacyOutcome.NOMINATED:
        # The one outcome that is a result. Votes and shares are not asserted
        # here — the results importer owns numbers — so this claim and the
        # imported vote counts are separate propositions that meet on the same
        # entities, each citing its own evidence.
        recorded.append(
            context.record(
                "contest_result",
                subject_id=person.entity_id,
                object_id=contest.entity_id,
                value={"won": True},
                excerpt=record.candidate.name,
            )
        )

    return Ingestion(
        subject_entity_id=person.entity_id,
        recorded=tuple(recorded),
        subject_created=person.created,
        related_entity_ids=(contest.entity_id,),
    )
