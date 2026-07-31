"""Open Civic Data divisions to JURISDICTION entities.

The highest-leverage import there is, and it writes no claims at all. What it
does is give every US jurisdiction an `ocd-division` identifier, which makes it
resolve at tier 0 in `resolve_entity_mention` forever after. Once Michigan is in
the graph under `ocd-division/country:us/state:mi`, an agent that says
"Michigan", a poll CSV that says "MI" and a results file that says "Michigan
(State of)" all land on the same entity — without any of them agreeing on a name
and without a reconciliation step that would have to be trusted.

It is also what `ContestKey` is built on. A contest key embeds a division ID, so
a race can only join to a jurisdiction if that jurisdiction exists under the ID
the key names.

**Scope.** `country-us.csv` is around 100,000 rows and goes down to precincts.
Importing all of it would take far longer and add six figures of entities nobody
queries. `DEFAULT_KINDS` keeps the levels that races are actually held at, and
the count of what was dropped is returned and logged — a filter that stays quiet
is indistinguishable from a source that was missing the data.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import re
from typing import ClassVar

from predictelection.importers.base import (
    FilteredParse,
    Importer,
    ImportRow,
    rows_from_delimited,
)
from predictelection.research.ingestion import Ingestion, IngestContext
from predictelection.research.scraped import ScrapedEntity
from predictelection.sql import EntityKind


OCD_DIVISIONS_URL = (
    "https://raw.githubusercontent.com/opencivicdata/ocd-division-ids/master/"
    "identifiers/country-us.csv"
)

DEFAULT_KINDS: tuple[str, ...] = (
    "country",
    "state",
    "district",
    "territory",
    "county",
    "cd",
    "sldu",
    "sldl",
    "place",
)
"""Division levels contests are actually held at.

`cd` is a congressional district, `sldu`/`sldl` the upper and lower state
legislative chambers. Deliberately excludes `precinct`, `ward`, `block` and the
rest: they are most of the file and no race is contested at that level.
"""

_ID_PATTERN = re.compile(r"^ocd-division/country:[a-z]{2}(?:/[a-z_]+:[^/]+)*$")


@dataclass(frozen=True, slots=True)
class OcdImporter(Importer):
    """One pass over the OCD division list.

    Idempotent by construction: every row resolves by its OCD ID, so a second
    run finds the entity the first one created and mints nothing.
    """

    name: ClassVar[str] = "import_ocd_divisions"
    media_type: ClassVar[str | None] = "text/csv"

    kinds: Sequence[str] = DEFAULT_KINDS
    url: str = OCD_DIVISIONS_URL

    @property
    def source_url(self) -> str:
        return self.url

    def parse(self, raw: bytes) -> FilteredParse:
        wanted = set(self.kinds)
        kept: list[ImportRow] = []
        skipped = 0

        for row in rows_from_delimited(raw):
            identifier = row.data.get("id", "")
            name = row.data.get("name", "")
            if not identifier or not name or not _ID_PATTERN.match(identifier):
                skipped += 1
                continue
            if _level_of(identifier) not in wanted:
                skipped += 1
                continue
            kept.append(row)

        return FilteredParse(rows=kept, skipped=skipped)

    def ingest(self, row: ImportRow, context: IngestContext) -> Ingestion:
        """Resolve the division, carrying its OCD ID. No claims.

        Deliberately claim-free: "Michigan is a jurisdiction called Michigan" is
        not a proposition anyone would want to review or supersede, and a claim
        needs a predicate that says something. The identifier *is* the useful
        output, and identifiers are entity metadata, not facts about the world.
        """

        # ScrapedEntity rather than a hand-built EntityMention: the
        # field-to-namespace mapping lives there, so nothing in this file has to
        # name the `ocd-division` namespace.
        resolved = context.resolve(
            EntityKind.JURISDICTION,
            ScrapedEntity(name=row.data["name"], ocd_id=row.data["id"]),
        )
        return Ingestion(
            subject_entity_id=resolved.entity_id,
            recorded=(),
            subject_created=resolved.created,
        )


def _level_of(identifier: str) -> str:
    """The type of the deepest segment: state, county, cd, place, precinct."""

    last = identifier.rsplit("/", 1)[-1]
    return last.split(":", 1)[0]


def divisions_in(raw: bytes, kinds: Sequence[str] = DEFAULT_KINDS) -> Iterator[str]:
    """Division IDs the importer would keep. For building contest keys offline."""

    for row in OcdImporter(kinds=kinds).parse(raw).rows:
        yield row.data["id"]
