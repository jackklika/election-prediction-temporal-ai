"""Contest identity, so three independent sources land on one CONTEST entity.

Phase 1 writes contests from an OCD importer, an FEC importer and an agent, none
of which can see the others. If identity is a name they fork immediately, and
that failure is already measured here: two runs on one subject produced 11 event
entities for 6 real debates, purely from re-phrased titles. A contest is worse
than an event, because "Michigan Governor 2026" and "2026 Michigan gubernatorial
election" and "MI-GOV 2026" are all reasonable and all different.

So identity is an identifier, derived from what a contest *is* rather than what
anyone calls it:

    ocd-division/country:us/state:mi/governor/2026/primary/democratic
    └──────────── division ────────┘ └office┘ cycle └stage┘ └─party─┘

Anything that can name the division, the office, the year and the stage arrives
at the same string, so `resolve_entity_mention` matches at tier 0 without
anything having to agree on wording.

The division carries the district, as OCD does — a House race is
`ocd-division/country:us/state:mi/cd:11` with office `us-house`, not office
`us-house-11`. Keeping geography in one place is what lets a contest join to the
jurisdiction the OCD importer created.

**A primary and a general are different contests.** They have different
candidates, different polls and different outcomes, and the stage segment is
what keeps them apart. A general has no party segment: it is a contest between
parties, not within one.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from predictelection.sql import ContestStage


CONTEST_KEY_NAMESPACE = "contest-key"
OFFICE_KEY_NAMESPACE = "office-key"
ELECTION_KEY_NAMESPACE = "election-key"

_DIVISION_PATTERN = re.compile(r"^ocd-division/[a-z0-9:_~.\-/]+$")
_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

CONTEST_KEY_PATTERN = (
    r"^ocd-division/[a-z0-9:_~.\-/]+"
    r"/[a-z0-9-]+/[0-9]{4}/(primary|runoff|general|special|caucus)"
    r"(/[a-z0-9-]+)?$"
)
"""Validation for a contest key arriving from outside, e.g. from a model.

Deliberately strict. A malformed key is rejected at the boundary rather than
minting a contest nobody else will ever resolve to; a wrong-but-well-formed key
is still possible, which is why the field description says to compute it rather
than guess it."""

EARLIEST_CYCLE = 1788
"""The first US presidential election. A cycle outside this range is a parse
error somewhere upstream, not a contest."""

LATEST_CYCLE = 2200


def normalize_slug(value: str) -> str:
    """Free text to a key segment: "US Senate" -> "us-senate".

    Offices and parties are slugs rather than enums because the set is open —
    state offices, ballot measures and party names vary by state, and an enum
    would force every new one through a schema change.
    """

    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"{value!r} has no usable characters")
    return slug


@dataclass(frozen=True, slots=True)
class ContestKey:
    """The deterministic identity of one contest."""

    division: str
    """An OCD division ID, which is also what the jurisdiction resolves by."""

    office: str
    cycle: int
    stage: ContestStage
    party: str | None = None
    """Whose primary. Must be absent for a general."""

    def __post_init__(self) -> None:
        if not _DIVISION_PATTERN.match(self.division):
            raise ValueError(f"not an OCD division ID: {self.division!r}")
        if not _SLUG_PATTERN.match(self.office):
            raise ValueError(f"office must be a slug, got {self.office!r}")
        if not EARLIEST_CYCLE <= self.cycle <= LATEST_CYCLE:
            raise ValueError(f"{self.cycle} is not a plausible election cycle")
        if self.party is not None and not _SLUG_PATTERN.match(self.party):
            raise ValueError(f"party must be a slug, got {self.party!r}")
        if self.party is not None and self.stage is ContestStage.GENERAL:
            # A general is contested between parties. A party-scoped general is
            # a primary that was labelled wrongly, and letting it through would
            # split the general into one contest per party.
            raise ValueError("a general election contest cannot be party-scoped")

    @classmethod
    def build(
        cls,
        *,
        division: str,
        office: str,
        cycle: int,
        stage: ContestStage,
        party: str | None = None,
    ) -> ContestKey:
        """Normalizing constructor, for callers holding human text."""

        return cls(
            division=division.strip().lower(),
            office=normalize_slug(office),
            cycle=cycle,
            stage=stage,
            party=normalize_slug(party) if party else None,
        )

    @classmethod
    def parse(cls, value: str) -> ContestKey:
        """Read a key back, so a stored identifier can be reasoned about.

        Splits from the right: the division itself contains slashes, and its
        length varies with how deep the division goes.
        """

        head, _, party_or_stage = value.rpartition("/")
        try:
            stage = ContestStage(party_or_stage)
        except ValueError:
            party = party_or_stage
            head, _, stage_text = head.rpartition("/")
            stage = ContestStage(stage_text)
        else:
            party = None

        head, _, cycle_text = head.rpartition("/")
        division, _, office = head.rpartition("/")
        if not division:
            raise ValueError(f"not a contest key: {value!r}")
        return cls(
            division=division,
            office=office,
            cycle=int(cycle_text),
            stage=stage,
            party=party,
        )

    def __str__(self) -> str:
        parts = [self.division, self.office, str(self.cycle), self.stage.value]
        if self.party is not None:
            parts.append(self.party)
        return "/".join(parts)

    @property
    def label(self) -> str:
        """A readable canonical_name for a newly minted contest.

        Only a label: identity is the key, so two sources disagreeing about
        wording still land on one entity and the first one to arrive names it.
        """

        who = f"{self.party.title()} " if self.party else ""
        office = self.office.replace("-", " ").title()
        return (
            f"{_division_label(self.division)} {office} "
            f"{self.cycle} {who}{self.stage.value.title()}"
        ).replace("  ", " ")

    @property
    def office_key(self) -> OfficeKey:
        """The seat being contested, independent of any one cycle."""

        return OfficeKey(division=self.division, office=self.office)

    @property
    def election_key(self) -> ElectionKey:
        """The election day this contest is decided on.

        Party drops out: Michigan's 2026 primary is one election containing
        every party's primary contests, not one election per party.
        """

        return ElectionKey(division=self.division, cycle=self.cycle, stage=self.stage)

    def at_stage(self, stage: ContestStage) -> ContestKey:
        """The same seat and cycle at a different stage.

        This is how `advances_to` gets written without anyone naming the general
        election: a primary derives its own successor. Party drops out, because
        a general is contested between parties rather than within one.
        """

        return ContestKey(
            division=self.division,
            office=self.office,
            cycle=self.cycle,
            stage=stage,
            party=None if stage is ContestStage.GENERAL else self.party,
        )


@dataclass(frozen=True, slots=True)
class OfficeKey:
    """A seat, derived the same way and for the same reason as a contest.

    `contest_for_office` is what makes "every governorship up in 2026"
    answerable, and it only works if two sources describing the Michigan
    governorship reach one OFFICE entity. Nobody issues office IDs either, so
    this is division plus office slug and nothing else — no cycle, because the
    seat outlives the election.
    """

    division: str
    office: str

    def __post_init__(self) -> None:
        if not _DIVISION_PATTERN.match(self.division):
            raise ValueError(f"not an OCD division ID: {self.division!r}")
        if not _SLUG_PATTERN.match(self.office):
            raise ValueError(f"office must be a slug, got {self.office!r}")

    def __str__(self) -> str:
        return f"{self.division}/{self.office}"

    @property
    def label(self) -> str:
        office = self.office.replace("-", " ").title()
        return f"{_division_label(self.division)} {office}"


@dataclass(frozen=True, slots=True)
class ElectionKey:
    """One election day in one place: the thing a contest is decided at.

    Several contests share one election — every seat on the same ballot — which
    is what `contest_of_election` is for, and what makes "everything decided on
    this date" a query rather than a date-range guess.
    """

    division: str
    cycle: int
    stage: ContestStage

    def __post_init__(self) -> None:
        if not _DIVISION_PATTERN.match(self.division):
            raise ValueError(f"not an OCD division ID: {self.division!r}")
        if not EARLIEST_CYCLE <= self.cycle <= LATEST_CYCLE:
            raise ValueError(f"{self.cycle} is not a plausible election cycle")

    def __str__(self) -> str:
        return f"{self.division}/{self.cycle}/{self.stage.value}"

    @property
    def label(self) -> str:
        where = _division_label(self.division)
        return f"{where} {self.cycle} {self.stage.value.title()} Election"


def _division_label(division: str) -> str:
    """A short human name for a division, for labelling a minted entity.

    Only cosmetic. Identity is the key, so the first source to arrive names the
    entity and later ones still resolve to it however they phrase things.
    """

    return division.rsplit("/", 1)[-1].split(":")[-1].upper()
