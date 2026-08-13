"""Donations: who funded whom, how much, and whether it was even support.

Written as the test of whether adding a domain is still cheap. It is: a
`PredicateSpec`, this file, three lines in `research/registry.py`, and — because
it is also worth researching rather than only importing — an agent and a
workflow. Nothing in `activities/`, `sql/ingest.py`, `query/` or `worker/`
changed, and no migration was needed, because a predicate is a data row rather
than an enum branch.

Two modelling choices are worth stating, because they are the ones a naive
version gets wrong:

**A donation is a claim about the donor.** The subject is whoever gave the
money. That keeps one shape for a person maxing out, a PAC bundling, and a super
PAC spending against a candidate — and it makes "who funded this campaign" a
query over objects rather than a special table.

**Support is not implied.** An independent expenditure can be spent *against*
its subject, and that is common enough that treating every donation as backing
would produce a confidently wrong funding picture. `DonationValue.supporting`
carries it, nullable, and the agent is told to leave it null rather than assume.

The recipient may be a person, a committee (an organization), or a contest —
money aimed at a ballot measure has no candidate to attach to. All three are
existing entity kinds, which is why this needed no schema change.
"""

from __future__ import annotations

from decimal import Decimal
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
    DonationKind,
    EntityKind,
    TimePrecision,
    Validity,
)


class ScrapedDonation(ScrapedRecord):
    """One contribution as a filing or a report states it."""

    record_type: Literal["donation"] = "donation"

    donor: ScrapedEntity = Field(
        description=(
            "Who gave the money, named exactly as the source writes them. A PAC "
            "is the donor, not the people who funded the PAC."
        )
    )
    donor_kind: EntityKind = Field(
        default=EntityKind.PERSON,
        description=(
            "Whether the donor is a person, an organization or a party. PACs, "
            "unions, super PACs and corporations are organizations."
        ),
    )
    recipient: ScrapedEntity = Field(
        description=(
            "Who received it or who it was spent about — a candidate, a "
            "committee, or a ballot measure's contest."
        )
    )
    recipient_kind: EntityKind = Field(
        default=EntityKind.PERSON,
        description=(
            "'person' for a candidate, 'organization' for a committee, "
            "'contest' for money aimed at a ballot measure."
        ),
    )

    amount: Decimal | None = Field(
        default=None,
        ge=0,
        description=(
            "The figure as published. Null when the source says money changed "
            "hands without stating how much — never estimate one."
        ),
    )
    currency: str = Field(default="USD", min_length=3, max_length=3)
    kind: DonationKind = Field(
        default=DonationKind.CONTRIBUTION,
        description=(
            "'contribution' when it went to the campaign, "
            "'independent_expenditure' when spent about them by someone acting "
            "independently, 'in_kind' for goods or services, 'loan' when "
            "repayable. These are legally different things, not synonyms."
        ),
    )
    supporting: bool | None = Field(
        default=None,
        description=(
            "For an independent expenditure only: true if spent for the "
            "recipient, false if against. Leave null for a direct contribution "
            "and whenever the source does not say — money spent against a "
            "candidate is common and assuming support would invert the fact."
        ),
    )
    purpose: str | None = Field(
        default=None,
        max_length=300,
        description="What the filing says it was for, if it says.",
    )

    # The race, so a donation is anchored rather than floating. Same block the
    # endorsement record carries, for the same reason: it gives the recipient a
    # contest to resolve against so funding and candidacy meet on one entity.
    division_id: str = Field(
        pattern=r"^ocd-division/country:[a-z]{2}(?:/[a-z_]+:[^/]+)*$",
        description="Open Civic Data ID of where the race is held.",
    )
    office: str = Field(min_length=1, max_length=100)
    cycle: int = Field(ge=1788, le=2200)
    stage: ContestStage
    party: str | None = Field(default=None, max_length=100)

    given_on: ScrapedDateTime | None = Field(
        default=None,
        description=(
            "The date of the contribution. Null when the source does not date "
            "it — an invented date is a false fact about timing."
        ),
    )
    given_precision: TimePrecision = Field(
        default=TimePrecision.DAY,
        description=(
            "How precisely the source gave it. 'month' for 'in July', "
            "'year' for a cycle total; do not promote a vague date to a day."
        ),
    )

    def validity(self) -> Validity:
        """A donation is an event, not a state, so it is a point in time.

        Unlike an endorsement, which holds over an interval and can be withdrawn,
        money given on a date stays given. A refund is a *new* record — a
        negative-signed filing is its own row in the source too — rather than an
        edit that makes the original disappear.
        """

        if self.given_on is None:
            return Validity.timeless()
        return Validity.on(self.given_on, self.given_precision)

    def contest_key(self) -> ContestKey:
        return ContestKey.build(
            division=self.division_id,
            office=self.office,
            cycle=self.cycle,
            stage=self.stage,
            party=self.party,
        )


def ingest_donation(record: ScrapedDonation, context: IngestContext) -> Ingestion:
    """One donation claim, plus the contest it was given in aid of.

    The recipient resolves against the contest entity when the money was aimed at
    a race rather than a person, which is what makes ballot-measure spending
    expressible without a second record type.
    """

    key = record.contest_key()
    contest = context.resolve(
        EntityKind.CONTEST,
        ScrapedEntity(name=key.label, contest_key=str(key)),
    )
    donor = context.resolve(record.donor_kind, record.donor)
    recipient = (
        contest
        if record.recipient_kind is EntityKind.CONTEST
        else context.resolve(record.recipient_kind, record.recipient)
    )

    # Every field, including the null ones. `DonationValue` validates and fills
    # its own defaults, so omitting a null here changes nothing about what is
    # stored — and reading as though it might invites a caller to treat "absent"
    # and "null" as different, when the payload only has one of them.
    value: dict[str, object] = {
        "amount": record.amount,
        "currency": record.currency.upper(),
        "kind": record.kind,
        "supporting": record.supporting,
        "purpose": record.purpose,
    }

    amount = f"${record.amount:,}" if record.amount is not None else "an unstated sum"
    recorded = context.record(
        "donated_to",
        subject_id=donor.entity_id,
        object_id=recipient.entity_id,
        value=value,
        validity=record.validity(),
        excerpt=f"{record.donor.name} → {record.recipient.name}, {amount}",
    )
    return Ingestion(
        subject_entity_id=donor.entity_id,
        recorded=(recorded,),
        subject_created=donor.created,
        related_entity_ids=(recipient.entity_id, contest.entity_id),
    )
