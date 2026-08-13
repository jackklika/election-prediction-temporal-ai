"""Derived identity: contests, offices, elections and events.

Named for contests because that is where it started; it now holds every key this
project derives rather than reads. Worth renaming if it grows again.

The shared argument, and the reason these are not names:

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
import datetime as dt
import re

# `normalize_slug` is imported rather than defined here. It moved down to
# `sql.entity`, beside `normalize_entity_name`, when the review queue needed it:
# `sql` is the bottom layer, and a reader there reaching up into `research` for a
# string function closed an import cycle. Re-exported so callers that think of it
# as "the thing that builds contest keys" can still get it from here.
from predictelection.sql import (
    ContestStage,
    PoliticalEventKind,
    TimePrecision,
    normalize_slug,
)


CONTEST_KEY_NAMESPACE = "contest-key"
OFFICE_KEY_NAMESPACE = "office-key"
ELECTION_KEY_NAMESPACE = "election-key"
EVENT_KEY_NAMESPACE = "event-key"

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


DAY_OR_FINER: frozenset[TimePrecision] = frozenset(
    {
        TimePrecision.DAY,
        TimePrecision.HOUR,
        TimePrecision.MINUTE,
        TimePrecision.SECOND,
        TimePrecision.EXACT,
    }
)
"""Precisions that pin an event to a calendar day.

A source that only said "September 2026" cannot key an event: the date is the
discriminator, and inventing one would merge every debate that month.
"""


@dataclass(frozen=True, slots=True)
class EventKey:
    """When and where an event happened, instead of what someone called it.

    This is the identity failure the whole data model roadmap opens with: two
    runs on one subject produced 11 event entities for 6 real debates, because
    the agent re-phrased each title — `(WOOD TV8` vs `(WOOD-TV`, `First` and
    `Second` prefixes, words reordered. Every other entity kind here now
    resolves on a derived key; events were the last one still resolving on a
    name, which is doubly odd given they are where the problem was measured.

    Date is the discriminator that names cannot supply: two debates a month
    apart share nearly all of their words, while the same debate described twice
    shares a date. `host` separates the rarer case of two events in one place on
    one day — without it they would merge, and wrongly merging two real events
    loses data in a way a fork does not.

    Deliberately built from the *resolved* jurisdiction's OCD division rather
    than from whatever the source called the place, and only when that division
    exists. A key that sometimes came from an OCD ID and sometimes from a name
    would fork on exactly the axis it is meant to fix, so no division means no
    key and the event falls back to resolving by title.
    """

    division: str
    kind: PoliticalEventKind
    date: dt.date
    host: str | None = None

    def __post_init__(self) -> None:
        if not _DIVISION_PATTERN.match(self.division):
            raise ValueError(f"not an OCD division ID: {self.division!r}")
        if self.host is not None and not _SLUG_PATTERN.match(self.host):
            raise ValueError(f"host must be a slug, got {self.host!r}")

    @classmethod
    def build(
        cls,
        *,
        division: str,
        kind: PoliticalEventKind,
        moment: dt.datetime,
        precision: TimePrecision,
        host: str | None = None,
    ) -> EventKey | None:
        """Normalizing constructor. None when the source was too vague to key."""

        if precision not in DAY_OR_FINER:
            return None
        return cls(
            division=division.strip().lower(),
            kind=kind,
            date=moment.date(),
            host=normalize_slug(host) if host else None,
        )

    @classmethod
    def parse(cls, value: str) -> EventKey:
        """Split from the right — the division contains slashes of its own."""

        head, _, tail = value.rpartition("/")
        try:
            date = dt.date.fromisoformat(tail)
            host = None
        except ValueError:
            host = tail
            head, _, date_text = head.rpartition("/")
            date = dt.date.fromisoformat(date_text)

        division, _, kind = head.rpartition("/")
        if not division:
            raise ValueError(f"not an event key: {value!r}")
        return cls(
            division=division,
            kind=PoliticalEventKind(kind),
            date=date,
            host=host,
        )

    def __str__(self) -> str:
        parts = [self.division, self.kind.value, self.date.isoformat()]
        if self.host is not None:
            parts.append(self.host)
        return "/".join(parts)

    @property
    def label(self) -> str:
        where = _division_label(self.division)
        who = f" ({self.host.replace('-', ' ').title()})" if self.host else ""
        return (
            f"{where} {self.kind.value.replace('_', ' ').title()} "
            f"{self.date.isoformat()}{who}"
        )


@dataclass(frozen=True, slots=True)
class PollKey:
    """Which poll this is, independent of who reported it.

    A poll has no published identifier — FiveThirtyEight, Wikipedia and
    Ballotpedia each describe "the EPIC-MRA poll of the Michigan Senate primary
    that finished fielding July 28" in their own words. Pollster, contest and
    fieldwork end date are the coordinates they all state, so they are the
    identity; everything else (numbers, sample, wording) is *content*, and
    content dedup is the payload hash one level down.

    Fieldwork end rather than publication date: publication varies by outlet
    (Wikipedia dates the release, an aggregator dates the entry), fieldwork is a
    property of the poll itself. A source that does not state fieldwork dates
    cannot key the poll — the caller falls back to an unkeyed Poll row and a
    ReviewTask, rather than this class inventing a date.
    """

    contest: ContestKey
    pollster: str
    fieldwork_end: dt.date

    def __post_init__(self) -> None:
        if not _SLUG_PATTERN.match(self.pollster):
            raise ValueError(f"pollster must be a slug, got {self.pollster!r}")

    @classmethod
    def build(
        cls, *, contest: ContestKey, pollster: str, fieldwork_end: dt.date
    ) -> PollKey:
        return cls(
            contest=contest,
            pollster=normalize_slug(pollster),
            fieldwork_end=fieldwork_end,
        )

    @classmethod
    def parse(cls, value: str) -> PollKey:
        """Contest first (variable segments), pollster and date at the end."""

        head, _, date_text = value.rpartition("/")
        contest_text, _, pollster = head.rpartition("/")
        if not contest_text:
            raise ValueError(f"not a poll key: {value!r}")
        return cls(
            contest=ContestKey.parse(contest_text),
            pollster=pollster,
            fieldwork_end=dt.date.fromisoformat(date_text),
        )

    def __str__(self) -> str:
        return f"{self.contest}/{self.pollster}/{self.fieldwork_end.isoformat()}"


def _division_label(division: str) -> str:
    """A short human name for a division, for labelling a minted entity.

    Only cosmetic. Identity is the key, so the first source to arrive names the
    entity and later ones still resolve to it however they phrase things.
    """

    return division.rsplit("/", 1)[-1].split(":")[-1].upper()
