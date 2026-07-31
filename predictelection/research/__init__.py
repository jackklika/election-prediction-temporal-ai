from predictelection.research.archive import SourceArchive
from predictelection.research.debates import (
    ScrapedDebate,
    ingest_debate,
)
from predictelection.research.ingestion import (
    IngestContext,
    Ingestion,
)
from predictelection.research.registry import (
    INGESTORS,
    Ingestor,
    ScrapedPayload,
    ingestor_for,
    payload_types,
)
from predictelection.research.scraped import (
    ScrapedEntity,
    ScrapedModel,
    ScrapedRecord,
)

__all__ = [
    "INGESTORS",
    "IngestContext",
    "Ingestion",
    "Ingestor",
    "ScrapedDebate",
    "ScrapedEntity",
    "ScrapedModel",
    "ScrapedPayload",
    "ScrapedRecord",
    "SourceArchive",
    "ingest_debate",
    "ingestor_for",
    "payload_types",
]
