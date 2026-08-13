"""Wikipedia race-article results tables to contest_result claims.

The first writer for `contest_result`, and deliberately a narrow one: it
asserts **votes and share, never victory**. The results table states counts;
it does not mark the winner, and the roadmap's rule stands — `won` is set
explicitly from a source that says so (the candidacy agent reads "Nominee"),
never inferred from place. The 2026 Wisconsin Democratic primary shows why
inference is a trap in both directions: the winner had previously withdrawn
from the race, and a withdrawn candidate out-polled two active ones.

Counts change — election night, canvass, certified. Re-importing an updated
page yields new claims with new values; the old ones stay citable against the
bytes that stated them, and `new_claim_supersession` links corrections when a
human confirms. Nothing here ever mutates a stored result.

Rows that are not a candidate's count — "Total votes", "Write-in", turnout —
are counted as skips, never guessed at. A "(withdrawn)" suffix on a name is
ballot metadata, not part of the name; it is stripped for resolution and kept
in the claim's excerpt so review sees what the page actually printed.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import ClassVar

from predictelection.importers.base import FilteredParse, Importer, ImportRow
from predictelection.importers.wikipedia_polls import _PageWalker, _section_for
from predictelection.research.contests import ContestKey
from predictelection.research.ingestion import Ingestion, IngestContext
from predictelection.research.polls import ScrapedEntity
from predictelection.sql import ContestStage, EntityKind


_NOT_A_CANDIDATE = re.compile(
    r"^(total|turnout|write-?in|majority|plurality|n/?a|others?)\b", re.IGNORECASE
)
_STATUS_SUFFIX = re.compile(r"\s*\((withdrawn|withdrew|deceased|disqualified)\)\s*$", re.I)


def parse_votes(text: str) -> int | None:
    cleaned = text.replace(",", "").strip()
    return int(cleaned) if cleaned.isdigit() else None


def parse_share(text: str) -> Decimal | None:
    cleaned = text.replace("%", "").strip()
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


@dataclass(frozen=True, slots=True)
class WikipediaResultsImporter(Importer):
    """Results tables from one race article, one contest per section."""

    name: ClassVar[str] = "import_wikipedia_results"
    media_type: ClassVar[str | None] = "text/html"

    url: str = ""
    division: str = ""
    office: str = ""
    cycle: int = 0

    @property
    def source_url(self) -> str:
        return self.url

    @property
    def subject(self) -> str:
        return f"{self.division} {self.office} {self.cycle} results"

    def parse(self, raw: bytes) -> FilteredParse:
        walker = _PageWalker()
        walker.feed(raw.decode("utf-8", errors="replace"))

        kept: list[ImportRow] = []
        skipped = 0
        index = 0
        for table in walker.tables:
            if not table.h3.lower().startswith("results"):
                continue
            section = _section_for(table.h2)
            if section is None or len(table.grid) < 2:
                skipped += 1
                continue
            stage, party = section
            columns = _result_columns(table.grid[0])
            if columns is None:
                skipped += 1
                continue
            name_col, votes_col, share_col = columns
            for row in table.grid[1:]:
                if len(row) <= max(votes_col, share_col):
                    skipped += 1
                    continue
                name = row[name_col].strip()
                votes = parse_votes(row[votes_col])
                share = parse_share(row[share_col])
                if not name or _NOT_A_CANDIDATE.match(name) or votes is None:
                    skipped += 1
                    continue
                kept.append(
                    ImportRow(
                        index=index,
                        data={
                            "stage": stage.value,
                            "party": party or "",
                            "printed_name": name,
                            "name": _STATUS_SUFFIX.sub("", name),
                            "votes": str(votes),
                            "share": str(share) if share is not None else "",
                        },
                    )
                )
                index += 1
        return FilteredParse(rows=kept, skipped=skipped)

    def ingest(self, row: ImportRow, context: IngestContext) -> Ingestion:
        data = row.data
        key = ContestKey.build(
            division=self.division,
            office=self.office,
            cycle=self.cycle,
            stage=ContestStage(data["stage"]),
            party=data["party"] or None,
        )
        person = context.resolve(EntityKind.PERSON, ScrapedEntity(name=data["name"]))
        contest = context.resolve(
            EntityKind.CONTEST, ScrapedEntity(name=key.label, contest_key=str(key))
        )
        value: dict[str, object] = {"votes": int(data["votes"])}
        if data["share"]:
            value["share"] = Decimal(data["share"])
        recorded = context.record(
            "contest_result",
            subject_id=person.entity_id,
            object_id=contest.entity_id,
            value=value,
            # The page's own wording, "(withdrawn)" included: review should see
            # exactly what the table printed next to the count.
            excerpt=f"{data['printed_name']}: {data['votes']} ({data['share'] or '?'}%)",
        )
        return Ingestion(
            subject_entity_id=person.entity_id,
            recorded=(recorded,),
            subject_created=person.created,
            related_entity_ids=(contest.entity_id,),
        )


def _result_columns(header: list[str]) -> tuple[int, int, int] | None:
    """(candidate, votes, share) positions, or None if not a results table."""

    lowered = [cell.lower() for cell in header]
    try:
        name = next(i for i, c in enumerate(lowered) if c.startswith("candidate"))
        votes = next(i for i, c in enumerate(lowered) if c.startswith("votes"))
        share = next(i for i, c in enumerate(lowered) if c.strip().startswith("%"))
    except StopIteration:
        return None
    return (name, votes, share)

