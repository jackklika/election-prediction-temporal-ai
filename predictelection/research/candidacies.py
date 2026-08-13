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
- **Nothing here writes `contest_result`.** Counts *and* the outcome come from
  the results importer, which reads both from the same page deterministically —
  the table for votes, the "Nominee" heading for `won`. Two writers of one
  predicate is how a graph acquires contradictions: an outcome claim saying
  "won" beside a count claim whose `won` defaulted to False.
- `outcome` is therefore not a claim. It earns its place by validating the
  stints — a withdrawal implies the last stint ended — and by telling review
  what the page called this candidacy.

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
from predictelection.research.scraped import (
    ScrapedDateTime,
    ScrapedEntity,
    ScrapedModel,
    ScrapedRecord,
)
from predictelection.sql import (
    ContestStage,
    EntityKind,
    TimePrecision,
    Validity,
)


class CandidacyOutcome(StrEnum):
    """How the candidacy ended, as the source states it."""

    NOMINATED = "nominated"
    """Won this contest. Recorded as `contest_result.won` by the results
    importer, which reads the same page's Nominee heading — not from here."""

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

    entered_on: ScrapedDateTime | None = Field(
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
    left_on: ScrapedDateTime | None = Field(
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
    def _stints_do_not_overlap(self):
        """Stints are consecutive periods, so they must not overlap.

        Overlapping stints would assert someone was simultaneously in the race
        twice, which usually means a re-entry was reported with the *original*
        announcement date instead of the re-entry date. Undated stints are
        skipped rather than rejected — a source that does not date a withdrawal
        is normal, and refusing it would lose the candidacy entirely.
        """

        previous_exit: datetime | None = None
        for stint in self.stints:
            if (
                previous_exit is not None
                and stint.entered_on is not None
                and stint.entered_on < previous_exit
            ):
                raise ValueError(
                    f"stint starting {stint.entered_on.date()} overlaps the "
                    f"previous one, which ended {previous_exit.date()}"
                )
            previous_exit = stint.left_on or previous_exit
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

    return Ingestion(
        subject_entity_id=person.entity_id,
        recorded=tuple(recorded),
        subject_created=person.created,
        related_entity_ids=(contest.entity_id,),
    )
