"""The list of things that can be scraped, and what turns each into claims.

This is the only file that has to change when a domain is added. The activity
that writes records, the contract that carries them over the wire, the workflow
that loops over them and the worker that registers all of it are domain-free by
construction and stay that way.

`ScrapedPayload` is a tagged union rather than the `ScrapedRecord` base class
because Temporal deserializes activity inputs through the Pydantic converter,
and validating a debate payload against the base class raises under
`ScrapedModel`'s `extra="forbid"`. The `record_type` discriminator is what lets
one wire contract carry any domain; Pydantic refuses to build the union at
import if a member forgets to declare one.

Adding a domain:

1. give the record a `record_type: Literal["thing"] = "thing"`
2. add it to `ScrapedPayload`
3. add its ingest function to `INGESTORS`

Nothing else. If a fourth step appears, the seam has leaked and belongs back
here rather than in the layer that leaked it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, get_args

from pydantic import Field

from predictelection.research.candidacies import ScrapedCandidacy, ingest_candidacy
from predictelection.research.debates import ScrapedDebate, ingest_debate
from predictelection.research.donations import ScrapedDonation, ingest_donation
from predictelection.research.endorsements import (
    ScrapedEndorsement,
    ingest_endorsement,
)
from predictelection.research.ingestion import Ingestion, IngestContext
from predictelection.research.polls import ScrapedPoll, ingest_poll
from predictelection.research.scraped import ScrapedRecord
from predictelection.research.structure import (
    ScrapedRaceStructure,
    ingest_race_structure,
)


ScrapedPayload = Annotated[
    ScrapedDebate
    | ScrapedRaceStructure
    | ScrapedPoll
    | ScrapedCandidacy
    | ScrapedEndorsement
    | ScrapedDonation,
    Field(discriminator="record_type"),
]
"""Every record shape an activity may be handed.

Removing a member breaks replay of workflows already in flight, which is the
same constraint every contract here carries. Adding one does not.
"""


Ingestor = Callable[[Any, IngestContext], Ingestion]
"""(record, context) -> Ingestion. Uniform so the registry can dispatch blind."""


INGESTORS: dict[type[ScrapedRecord], Ingestor] = {
    ScrapedDebate: ingest_debate,
    ScrapedRaceStructure: ingest_race_structure,
    ScrapedPoll: ingest_poll,
    ScrapedCandidacy: ingest_candidacy,
    ScrapedEndorsement: ingest_endorsement,
    ScrapedDonation: ingest_donation,
}


def ingestor_for(record: ScrapedRecord) -> Ingestor:
    """The ingest function for a record, by its concrete type.

    Raises rather than defaulting: a record with no ingestor is a domain that
    was half-registered, and silently recording nothing for it would look
    exactly like a source that mentioned nothing.
    """

    try:
        return INGESTORS[type(record)]
    except KeyError:
        raise LookupError(
            f"{type(record).__name__} is in ScrapedPayload but not in INGESTORS"
        ) from None


def payload_types() -> tuple[type[ScrapedRecord], ...]:
    """The union's members, for the test that keeps it and INGESTORS in step."""

    members = get_args(ScrapedPayload)[0]
    return get_args(members) or (members,)
