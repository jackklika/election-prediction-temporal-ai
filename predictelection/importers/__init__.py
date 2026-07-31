"""Deterministic ingestion from published files.

The counterpart to `predictelection/research/`, which handles what agents read.
Same write path, same provenance chain, no LLM: an importer archives its source
file exactly as a scrape archives a web page, and cites a row inside it rather
than a paragraph.

Use one whenever the data exists as a feed. Vote counts and poll percentages
from a model are a needless risk when a CSV is published.
"""

from predictelection.importers.base import (
    FilteredParse,
    ImportResult,
    ImportRow,
    Importer,
    rows_from_delimited,
    run_import,
)
from predictelection.importers.fec import FecCandidateImporter
from predictelection.importers.ocd import OcdImporter

__all__ = [
    "FecCandidateImporter",
    "FilteredParse",
    "ImportResult",
    "ImportRow",
    "Importer",
    "OcdImporter",
    "rows_from_delimited",
    "run_import",
]
