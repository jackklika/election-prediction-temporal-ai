"""Shared assertions for ingestion tests.

Every predicate that gets a writer needs a test proving re-ingestion does not
duplicate — it is the single most common way to break the graph, and it is easy
to write a version that passes for the wrong reason. Counting only claims misses
the failure where the evidence anchor changes and every assertion is written
twice; counting only entities misses everything.

So the check lives here, counts all five tables, and is one line per domain.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection.sql import (
    Claim,
    ClaimAssertion,
    Entity,
    EvidenceAnchor,
    ReviewTask,
)


@dataclass(frozen=True, slots=True)
class GraphCounts:
    """Everything an ingestion can add, so nothing can grow unnoticed."""

    entities: int
    claims: int
    assertions: int
    evidence_anchors: int
    review_tasks: int

    def describe(self, other: GraphCounts) -> str:
        return ", ".join(
            f"{field}: {getattr(other, field)} -> {getattr(self, field)}"
            for field in self.__slots__
            if getattr(self, field) != getattr(other, field)
        )


def graph_counts(session: Session) -> GraphCounts:
    def count(model) -> int:
        return session.scalar(select(func.count(model.id))) or 0

    return GraphCounts(
        entities=count(Entity),
        claims=count(Claim),
        assertions=count(ClaimAssertion),
        evidence_anchors=count(EvidenceAnchor),
        review_tasks=count(ReviewTask),
    )


def assert_reingestion_is_idempotent(
    session: Session, ingest: Callable[[], object]
) -> None:
    """Run the ingestion twice; nothing may move the second time.

    The second run is what a Temporal retry does, and what a nightly re-import
    of an unchanged file does. Both must be able to write nothing.

    Note what this catches beyond duplicate claims: a new EvidenceAnchor on the
    second pass means each claim gained a second assertion citing "the same
    bytes, observed again". That is correct for a genuine re-observation and
    wrong for a retry, and the claim count alone cannot tell them apart.
    """

    ingest()
    session.flush()
    before = graph_counts(session)

    ingest()
    session.flush()
    after = graph_counts(session)

    assert after == before, f"re-ingestion changed the graph — {after.describe(before)}"
