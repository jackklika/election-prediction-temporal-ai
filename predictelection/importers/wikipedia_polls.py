"""Wikipedia race-article poll tables to ScrapedPoll records.

Tier 0 of the poll pipeline: everything here is deterministic. Numbers travel
from `<td>` to database untouched by any model — the roadmap's one absolute —
and every judgment this file cannot make deterministically is refused loudly
(a counted, logged row failure) rather than guessed. The agent's role, when it
arrives, is normalizing the refusals, not replacing the parser.

What the real page taught (2026 Michigan Senate, fetched 2026-08-12):

- **Tables must be recognized by header, not position.** The first table under
  a "Polling" heading is often an *aggregation* table ("Source of poll
  aggregation" — 270toWin, DDHQ). Averages are derived, not primary; importing
  them as polls would double-count every poll they summarize. Recognition is
  by header shape, and unrecognized tables are counted, never silently passed.
- **Text must come from a real parser.** MediaWiki cells carry `data-mw`
  attributes containing escaped HTML; regex tag-stripping leaks it into cell
  text. `html.parser` never treats attributes as text.
- **Footnote refs are elements, not string suffixes.** `<sup class="mw-ref">`
  subtrees are skipped during text extraction, so "437 (LV)[j]" arrives as
  "437 (LV)".
- **`—` is structure, not data.** An em-dash cell means the candidate was not
  offered in that poll's question. The reading is omitted, not zero.
- **rowspan is load-bearing.** One poll's pollster/date cells span its LV and
  RV rows. The grid is expanded before any row is read.

The section a table sits under supplies what its rows cannot: an h2 like
"Democratic primary" maps to the contest's stage and party, completing the
`ContestKey` that attaches every poll to the contest the FEC import created.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
import datetime as dt
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
import json
import logging
import re
from typing import ClassVar

from predictelection.importers.base import FilteredParse, Importer, ImportRow
from predictelection.importers.normalize import CellNormalizer
from predictelection.research.contests import ContestKey
from predictelection.research.ingestion import Ingestion, IngestContext
from predictelection.research.polls import (
    PollReading,
    PollSampleReadings,
    ScrapedPoll,
    ingest_poll,
)
from predictelection.research.scraped import ScrapedEntity
from predictelection.sql import ContestStage, RecordOrigin


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML: headings, tables, and clean cell text


_SKIPPED_SUBTREES = frozenset({"style", "script"})
_REF_MARKER = "reference"
"""MediaWiki footnote sups carry class="mw-ref reference"."""


@dataclass
class SectionTable:
    """One wikitable and the headings it sits under."""

    h2: str
    h3: str
    grid: list[list[str]]
    """Rows of cell text, rowspan/colspan already expanded."""


class _PageWalker(HTMLParser):
    """One pass over the page: heading context plus expanded table grids.

    Stream-oriented because the page is 1.7 MB and only the tables matter.
    Nested tables mark the outer one malformed rather than corrupting its grid.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[SectionTable] = []
        self._h2 = ""
        self._h3 = ""
        self._heading_level: str | None = None
        self._heading_text: list[str] = []
        self._table_depth = 0
        self._malformed = False
        self._grid: dict[tuple[int, int], str] = {}
        self._row = -1
        self._column = 0
        self._cell_text: list[str] | None = None
        self._cell_span: tuple[int, int] = (1, 1)
        self._skip_depth = 0

    # -- headings ----------------------------------------------------------

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if self._skip_depth:
            self._skip_depth += 1
            return
        if tag in _SKIPPED_SUBTREES or (
            tag == "sup" and _REF_MARKER in (attributes.get("class") or "")
        ):
            self._skip_depth = 1
            return

        if tag == "br":
            # A line break is a space, wherever it appears: "Margin<br>of error"
            # must read "margin of error" or header recognition silently fails.
            if self._cell_text is not None:
                self._cell_text.append(" ")
            elif self._heading_level is not None:
                self._heading_text.append(" ")
            return

        if tag in ("h2", "h3"):
            self._heading_level = tag
            self._heading_text = []
        elif tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._grid, self._row, self._malformed = {}, -1, False
            else:
                self._malformed = True
        elif self._table_depth == 1:
            if tag == "tr":
                self._row += 1
                self._column = 0
            elif tag in ("td", "th"):
                while (self._row, self._column) in self._grid:
                    self._column += 1
                self._cell_text = []
                self._cell_span = (
                    int(attributes.get("rowspan") or 1),
                    int(attributes.get("colspan") or 1),
                )

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in ("h2", "h3") and self._heading_level == tag:
            text = " ".join("".join(self._heading_text).split())
            if tag == "h2":
                self._h2, self._h3 = text, ""
            else:
                self._h3 = text
            self._heading_level = None
        elif tag in ("td", "th") and self._cell_text is not None:
            text = " ".join("".join(self._cell_text).split())
            rows, cols = self._cell_span
            for dr in range(rows):
                for dc in range(cols):
                    self._grid.setdefault((self._row + dr, self._column + dc), text)
            self._cell_text = None
        elif tag == "table":
            self._table_depth -= 1
            if self._table_depth == 0 and not self._malformed and self._grid:
                self.tables.append(
                    SectionTable(h2=self._h2, h3=self._h3, grid=self._as_rows())
                )

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._heading_level is not None:
            self._heading_text.append(data)
        elif self._cell_text is not None:
            self._cell_text.append(data)

    def _as_rows(self) -> list[list[str]]:
        rows = max(r for r, _ in self._grid) + 1
        cols = max(c for _, c in self._grid) + 1
        return [[self._grid.get((r, c), "") for c in range(cols)] for r in range(rows)]


# ---------------------------------------------------------------------------
# Strict field parsers. Refusal is None; the caller decides what None costs.


_MONTHS = {
    month: i
    for i, month in enumerate(
        "january february march april may june july august september "
        "october november december".split(),
        start=1,
    )
}
_MONTHS |= {month[:3]: i for month, i in list(_MONTHS.items())}
_MONTHS["sept"] = 9
"""Abbreviated months appear on real pages ("Aug 24–29, 2023" — CA 2024)."""
_DASHES = re.compile(r"\s*[–—−-]\s*")
_DAY = re.compile(r"^(?:([a-z]+)\s+)?(\d{1,2})(?:,\s*(\d{4}))?$")
_ABSENT = frozenset({"", "—", "–", "-", "n/a", "tba"})


def parse_fieldwork(text: str) -> tuple[dt.date, dt.date] | None:
    """ "July 28–30, 2026" / "April 11–13, 2026" / "June 29 – July 29, 2026".

    Each side states what the other omits: a same-month range carries the month
    on the *left* ("April 11–13, 2026") and the year on the right; a cross-month
    range states the month twice. Anything that leaves a side without a month or
    year — "through July 30", a bare month — is a refusal, never an
    approximation: fieldwork dates are the poll's identity, and an invented day
    would merge or split polls.
    """

    cleaned = " ".join(text.split()).lower().replace(".", "")
    if cleaned in _ABSENT:
        return None
    parts = _DASHES.split(cleaned)
    if len(parts) > 2:
        return None

    pieces = [_DAY.match(part.strip()) for part in parts]
    if any(piece is None for piece in pieces):
        return None
    sides = [
        (
            _MONTHS.get(matched.group(1)) if matched.group(1) else None,
            int(matched.group(2)),
            int(matched.group(3)) if matched.group(3) else None,
        )
        for matched in pieces  # type: ignore[union-attr]
        if matched is not None
    ]
    if any(month is None for month, _, _ in sides) and all(
        month is None for month, _, _ in sides
    ):
        return None

    left = sides[0]
    right = sides[-1]
    end_month = right[0] or left[0]
    end_year = right[2] or left[2]
    if end_month is None or end_year is None:
        return None
    try:
        end = dt.date(end_year, end_month, right[1])
        start = dt.date(left[2] or end_year, left[0] or end_month, left[1])
    except ValueError:
        return None
    if start > end:  # "December 28 – January 3, 2026" crosses a year boundary
        start = start.replace(year=start.year - 1)
    return (start, end)


_SAMPLE = re.compile(r"^([\d,]+)?\s*\(?\s*([a-z]+)?\s*\)?$")


def parse_sample(text: str) -> tuple[int | None, str] | None:
    """ "437 (LV)" -> (437, "lv"). Absent is fine; unreadable is a refusal.

    The tolerant edges come from real pages: "905 LV" without parentheses is
    the 2018 convention, "3,045 LV)" is a hand-edited typo, and "– (LV)" states
    the population while omitting the count. An approximate count ("~329") is
    still a refusal — deciding that "approximately 329" may be stored as 329 is
    an interpretation, and interpretations belong to the normalizer, where they
    are marked as such.
    """

    cleaned = " ".join(text.split()).lower()
    if cleaned in _ABSENT:
        return (None, "unknown")
    for absent in sorted(_ABSENT - {""}, key=len, reverse=True):
        if cleaned.startswith(absent):
            cleaned = cleaned[len(absent) :].strip()
            break
    matched = _SAMPLE.match(cleaned)
    if not matched or not any(matched.groups()):
        return None
    size, population = matched.groups()
    return (
        int(size.replace(",", "")) if size else None,
        population or "unknown",
    )


def parse_margin(text: str) -> Decimal | None | tuple[()]:
    """ "± 4.7%" -> Decimal. Returns () on refusal — None means legitimately absent."""

    cleaned = " ".join(text.split()).replace("±", "").replace("%", "").strip()
    if cleaned.lower() in _ABSENT:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return ()


def parse_percentage(text: str) -> Decimal | None:
    """ "39%" -> Decimal("39"). Absent forms return None; junk raises ValueError.

    "<1%" is deliberately a refusal: it is a real datum this importer cannot
    represent exactly, and rounding it to 0 or 1 would be inventing a number.
    """

    cleaned = " ".join(text.split()).replace("%", "").strip()
    if cleaned.lower() in _ABSENT:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation as error:
        raise ValueError(f"unreadable percentage {text!r}") from error


# ---------------------------------------------------------------------------
# Recognizing and reading poll tables


_META_COLUMNS: tuple[tuple[str, str], ...] = (
    ("pollster", "poll source"),
    ("dates", "date"),
    ("sample", "sample"),
    ("moe", "margin of error"),
)
"""Header-prefix recognition, in expected order. "Source of poll aggregation"
fails the first prefix, which is what keeps averages out."""

_SECTIONS: tuple[tuple[str, ContestStage, str | None], ...] = (
    # Most specific first: "Republican primary runoff" must not stop at the
    # "republican primary" needle, or a runoff imports as its own primary.
    ("democratic primary runoff", ContestStage.RUNOFF, "democratic"),
    ("republican primary runoff", ContestStage.RUNOFF, "republican"),
    ("democratic primary", ContestStage.PRIMARY, "democratic"),
    ("republican primary", ContestStage.PRIMARY, "republican"),
    ("general election", ContestStage.GENERAL, None),
    # Top-two and blanket systems: one primary, all parties on one ballot, so
    # the contest is stage=primary with no party — the same shape Alaska and
    # California actually vote in ("Primary elections", "Nonpartisan blanket
    # primary"). Matched after the party-scoped needles so a party primary
    # never falls through to the partyless mapping.
    ("nonpartisan blanket primary", ContestStage.PRIMARY, None),
    ("nonpartisan primary", ContestStage.PRIMARY, None),
    ("primary election", ContestStage.PRIMARY, None),
    # A bare "Runoff election" h2 is a general-election runoff (Georgia 2022).
    ("runoff", ContestStage.RUNOFF, None),
)


def _classify(grid: list[list[str]]) -> dict[str, int] | None:
    """Column positions when this is an individual-polls table, else None."""

    if not grid or len(grid) < 2:
        return None
    header = [cell.lower() for cell in grid[0]]
    positions: dict[str, int] = {}
    cursor = 0
    for name, prefix in _META_COLUMNS:
        found = next(
            (i for i in range(cursor, len(header)) if header[i].startswith(prefix)),
            None,
        )
        if found is None:
            return None
        positions[name] = found
        cursor = found + 1
    return positions


def _section_for(h2: str) -> tuple[ContestStage, str | None] | None:
    lowered = h2.lower()
    for needle, stage, party in _SECTIONS:
        if needle in lowered:
            return (stage, party)
    return None


@dataclass(frozen=True, slots=True)
class WikipediaPollsImporter(Importer):
    """Poll tables from one race article.

    Parameterized by the `ContestKey` components the page cannot state —
    division, office, cycle — exactly as the FEC importer is parameterized by
    cycle. The section headings supply stage and party per table.
    """

    name: ClassVar[str] = "import_wikipedia_polls"
    media_type: ClassVar[str | None] = "text/html"

    url: str = ""
    division: str = ""
    office: str = ""
    cycle: int = 0
    contest_names: dict[tuple[str, str | None], str] = field(default_factory=dict)
    """Optional display names per (stage, party); the key label is the default."""

    normalizer: CellNormalizer | None = None
    """Tier 1: rewrites cells the strict parsers refuse. None — the default and
    what CI runs — keeps the importer fully deterministic; refusals then fail
    their row visibly instead of being smoothed."""

    @property
    def source_url(self) -> str:
        return self.url

    @property
    def subject(self) -> str:
        return f"{self.division} {self.office} {self.cycle} polls"

    def parse(self, raw: bytes) -> FilteredParse:
        walker = _PageWalker()
        walker.feed(raw.decode("utf-8", errors="replace"))

        # Grouped by (section, pollster, dates) across every table in the
        # section, not per table. Both layers matter: rowspanned LV/RV rows
        # within a table are one poll's samples, and the general election's
        # many head-to-head matchup tables are one poll's *questions* — the
        # same survey asks Stevens-vs-Rogers and McMorrow-vs-Rogers. Grouped
        # per table, those collide on one PollKey and are stored as
        # "disagreeing sources"; the first live run filed 61 spurious reviews
        # this way for a page containing 62 polls.
        GroupKey = tuple[str, str, str, str]  # stage, party, pollster, dates
        groups: dict[GroupKey, list[dict[str, str]]] = {}
        order: list[GroupKey] = []
        skipped = 0

        for table in walker.tables:
            if not table.h3.lower().startswith("polling"):
                continue
            section = _section_for(table.h2)
            columns = _classify(table.grid)
            if section is None or columns is None:
                # An aggregation table, or polling under a section this page
                # cannot map to a contest. Counted: a quiet filter here would
                # read as "Wikipedia had no polls".
                skipped += 1
                continue
            stage, party = section
            readings = [
                (i, label)
                for i, label in enumerate(table.grid[0])
                if i not in columns.values() and label.strip()
            ]
            for row in table.grid[1:]:
                if len(row) < len(table.grid[0]):
                    skipped += 1
                    continue
                if len({cell for cell in row if cell}) <= 2:
                    # A full-width colspan — "McMorrow suspends her campaign" —
                    # smeared across every column by the span expansion. An
                    # event marker, not a poll.
                    skipped += 1
                    continue
                sample_block = {
                    "sample": row[columns["sample"]],
                    "moe": row[columns["moe"]],
                }
                for position, label in readings:
                    sample_block[f"reading:{label}"] = row[position]
                key = (
                    stage.value,
                    party or "",
                    row[columns["pollster"]],
                    row[columns["dates"]],
                )
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(sample_block)

        kept = [
            ImportRow(
                index=index,
                data={
                    "stage": stage,
                    "party": party,
                    "pollster": pollster,
                    "dates": dates,
                    "samples": json.dumps(groups[(stage, party, pollster, dates)]),
                },
            )
            for index, (stage, party, pollster, dates) in enumerate(order)
        ]
        return FilteredParse(rows=kept, skipped=skipped)

    def ingest(self, row: ImportRow, context: IngestContext) -> Ingestion:
        data = row.data
        stage = ContestStage(data["stage"])
        party = data["party"] or None
        normalized = False

        fieldwork = parse_fieldwork(data["dates"])
        if fieldwork is None:
            fieldwork = self._retry(parse_fieldwork, "fieldwork_dates", data["dates"])
            normalized = True

        samples = []
        for block in json.loads(data["samples"]):
            sample = parse_sample(block["sample"])
            if sample is None:
                sample = self._retry(parse_sample, "sample", block["sample"])
                normalized = True
            margin = parse_margin(block["moe"])
            if isinstance(margin, tuple):  # the refusal sentinel; None is absent
                margin = self._retry(_margin_or_none, "margin", block["moe"])
                normalized = True
            readings = []
            for key, value in block.items():
                if not key.startswith("reading:"):
                    continue
                percentage = parse_percentage(value)  # raises on junk
                if percentage is None:
                    continue  # "—": not offered in this poll's question
                readings.append(
                    PollReading(
                        label=key.removeprefix("reading:"), percentage=percentage
                    )
                )
            if not readings:
                raise ValueError("no readable percentages in sample row")
            samples.append(
                PollSampleReadings(
                    population=sample[1],
                    sample_size=sample[0],
                    margin_of_error=margin,
                    readings=tuple(readings),
                )
            )

        contest_key = ContestKey.build(
            division=self.division,
            office=self.office,
            cycle=self.cycle,
            stage=stage,
            party=party,
        )
        poll = ScrapedPoll(
            source_url=self.url,
            pollster=data["pollster"],
            contest=ScrapedEntity(
                name=self.contest_names.get((stage.value, party), contest_key.label),
                contest_key=str(contest_key),
            ),
            fieldwork_started_on=fieldwork[0],
            fieldwork_ended_on=fieldwork[1],
            samples=tuple(samples),
        )
        if normalized:
            # A model interpreted at least one cell of this poll. The numbers
            # are still parsed, but the revision must say a model was involved:
            # review triages an interpreted date differently from a parsed one.
            context = replace(context, origin=RecordOrigin.MODEL)
        return ingest_poll(poll, context)

    def _retry(self, parser, kind: str, text: str):
        """One tier-1 pass: rewrite, then hold the rewrite to the same standard.

        The model's output is not a value — it is text the strict parser must
        still accept. No normalizer, or a rewrite that still refuses, and the
        original refusal stands as this row's failure.
        """

        if self.normalizer is None:
            raise ValueError(f"unreadable {kind} {text!r}")
        rewritten = self.normalizer.rewrite(kind, text)
        parsed = parser(rewritten) if rewritten is not None else None
        if parsed is None:
            raise ValueError(
                f"unreadable {kind} {text!r} (rewrite {rewritten!r} also refused)"
            )
        return parsed


def _margin_or_none(text: str):
    """parse_margin with the refusal sentinel folded to None, for _retry."""

    margin = parse_margin(text)
    return None if isinstance(margin, tuple) else margin


def poll_rows_in(raw: bytes, importer: WikipediaPollsImporter) -> Iterator[ImportRow]:
    """The rows the importer would ingest, for checking a page without a database."""

    yield from importer.parse(raw).rows
