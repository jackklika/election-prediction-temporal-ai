"""FEC bulk candidate data to candidacies and party affiliations.

The candidate master file (`cn.txt`) is pipe-delimited, headerless, and one row
per candidate per cycle. It gives a person, an FEC ID, a party, an office, a
state and a district — everything about a federal candidacy except the contest.

**The FEC publishes no contest rows.** There is no ID for "the 2026 Michigan
Senate general election", so a contest has to be synthesised, and how it is
synthesised decides whether this importer and the structure agent are talking
about the same race. That is what `ContestKey` is for: office, cycle, stage and
division combine into an identifier both sides derive independently. Without it
this file would mint "MI Senate 2026" and the agent would mint "2026 United
States Senate election in Michigan", and nothing would ever join them.

**Every candidate here is in a primary, not a general.** `cn.txt` says a person
filed for an office in a cycle under a party; it does not say they won the
nomination. Recording them in the general would assert something the source does
not support — and would put every party's candidates in one contest, which is
exactly the primary/general collapse the data model forbids. The general is a
separate CONTEST entity, linked later by `advances_to` once results exist.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import ClassVar

from predictelection.importers.base import (
    FilteredParse,
    Importer,
    ImportRow,
    rows_from_delimited,
)
from predictelection.research.contests import ContestKey
from predictelection.research.ingestion import Ingestion, IngestContext
from predictelection.research.scraped import ScrapedEntity
from predictelection.sql import ContestStage, EntityKind


CANDIDATE_MASTER_COLUMNS: tuple[str, ...] = (
    "CAND_ID",
    "CAND_NAME",
    "CAND_PTY_AFFILIATION",
    "CAND_ELECTION_YR",
    "CAND_OFFICE_ST",
    "CAND_OFFICE",
    "CAND_OFFICE_DISTRICT",
    "CAND_ICI",
    "CAND_STATUS",
    "CAND_PCC",
    "CAND_ST1",
    "CAND_ST2",
    "CAND_CITY",
    "CAND_ST",
    "CAND_ZIP",
)
"""Column order from the FEC's candidate master documentation.

The file ships without a header, so these names come from the spec rather than
the data. If the FEC reorders columns this list is the single thing to change —
and the parse will produce visible nonsense rather than silently misattribute a
party, because CAND_OFFICE only ever holds H, S or P.
"""

OFFICES: dict[str, str] = {"H": "us-house", "S": "us-senate", "P": "president"}

PARTIES: dict[str, str] = {
    "DEM": "democratic",
    "REP": "republican",
    "LIB": "libertarian",
    "GRE": "green",
    "IND": "independent",
    "NON": "nonpartisan",
}
"""FEC party codes worth normalizing. Anything else keeps its code, lowercased:
the list is long, mostly one-offs, and a slug of the code is still stable."""

ACTIVE_STATUSES = frozenset({"C", "N"})
"""Statutory candidate, or a new filer. Excludes P (prior cycle), F and other
withdrawn or future filings — those are not candidacies in this cycle."""


def default_candidate_url(cycle: int) -> str:
    """The FEC's per-cycle bulk file. Cycles are the even year of the election."""

    return f"https://www.fec.gov/files/bulk-downloads/{cycle}/cn{str(cycle)[-2:]}.zip"


@dataclass(frozen=True, slots=True)
class FecCandidateImporter(Importer):
    """One cycle of federal candidacies.

    Writes `candidate_in` and `party_affiliation`, and gives every candidate an
    `fec` identifier — which is what lets a later poll or results import that
    only knows the FEC ID land on the same person.
    """

    name: ClassVar[str] = "import_fec_candidates"
    media_type: ClassVar[str | None] = "text/plain"

    cycle: int
    url: str | None = None
    offices: frozenset[str] = field(default=frozenset(OFFICES))

    @property
    def source_url(self) -> str:
        return self.url or default_candidate_url(self.cycle)

    @property
    def subject(self) -> str:
        return f"federal candidates {self.cycle}"

    def parse(self, raw: bytes) -> FilteredParse:
        kept: list[ImportRow] = []
        skipped = 0

        for row in rows_from_delimited(
            raw, fieldnames=CANDIDATE_MASTER_COLUMNS, delimiter="|"
        ):
            if not _usable(row, self.offices, self.cycle):
                skipped += 1
                continue
            kept.append(row)

        return FilteredParse(rows=kept, skipped=skipped)

    def ingest(self, row: ImportRow, context: IngestContext) -> Ingestion:
        key = _contest_key_for(row, self.cycle)
        person = context.resolve(
            EntityKind.PERSON,
            ScrapedEntity(name=row.data["CAND_NAME"], fec_id=row.data["CAND_ID"]),
        )
        contest = context.resolve(
            EntityKind.CONTEST,
            ScrapedEntity(name=key.label, contest_key=str(key)),
        )
        recorded = [
            context.record(
                "candidate_in",
                subject_id=person.entity_id,
                object_id=contest.entity_id,
                excerpt=row.data["CAND_NAME"],
            )
        ]

        # Party is a claim about the person, separate from the contest they are
        # standing in, because it outlives any one race and can change.
        party_code = row.data.get("CAND_PTY_AFFILIATION", "")
        if party_code:
            party = context.resolve(
                EntityKind.PARTY, ScrapedEntity(name=_party_slug(party_code))
            )
            recorded.append(
                context.record(
                    "party_affiliation",
                    subject_id=person.entity_id,
                    object_id=party.entity_id,
                    excerpt=party_code,
                )
            )

        return Ingestion(
            subject_entity_id=person.entity_id,
            recorded=tuple(recorded),
            subject_created=person.created,
            related_entity_ids=(contest.entity_id,),
        )


def _usable(row: ImportRow, offices: frozenset[str], cycle: int) -> bool:
    data = row.data
    if data.get("CAND_ID", "").upper() == "CAND_ID":
        return False  # a header row, if a future file gains one
    if data.get("CAND_OFFICE") not in offices:
        return False
    if data.get("CAND_STATUS") not in ACTIVE_STATUSES:
        return False
    if not data.get("CAND_NAME") or not data.get("CAND_ID"):
        return False
    return data.get("CAND_ELECTION_YR", "").strip() == str(cycle)


def _party_slug(code: str) -> str:
    return PARTIES.get(code.upper(), code.strip().lower())


def _contest_key_for(row: ImportRow, cycle: int) -> ContestKey:
    """Synthesise the contest this candidacy is for.

    The district goes in the division, matching OCD, so a House contest joins to
    the congressional-district jurisdiction rather than to a district that only
    exists inside an office name.
    """

    office_code = row.data["CAND_OFFICE"]
    state = row.data.get("CAND_OFFICE_ST", "").strip().lower()

    if office_code == "P":
        division = "ocd-division/country:us"
    elif not state:
        raise ValueError(f"{row.data['CAND_ID']} has no office state")
    elif office_code == "H":
        district = (row.data.get("CAND_OFFICE_DISTRICT") or "").strip().lstrip("0")
        # At-large states file district 00; OCD writes those as cd:1.
        division = f"ocd-division/country:us/state:{state}/cd:{district or '1'}"
    else:
        division = f"ocd-division/country:us/state:{state}"

    return ContestKey.build(
        division=division,
        office=OFFICES[office_code],
        cycle=cycle,
        # Filing for an office is not winning the nomination, so this is the
        # party's primary. The general is a separate contest, joined later by
        # advances_to once a nominee is known.
        stage=ContestStage.PRIMARY,
        party=_party_slug(row.data.get("CAND_PTY_AFFILIATION", "") or "independent"),
    )


def candidacies_in(raw: bytes, cycle: int) -> Iterator[tuple[str, ContestKey]]:
    """(FEC ID, contest key) pairs, for checking a parse without a database."""

    importer = FecCandidateImporter(cycle=cycle)
    for row in importer.parse(raw).rows:
        yield row.data["CAND_ID"], _contest_key_for(row, cycle)
