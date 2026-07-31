"""Invariants over the supersession and redirect graphs.

Every one of these chains has a CHECK forbidding self-reference, but a two-step
cycle satisfies all of them: claim_supersession rows A->B and B->A each have a
distinct predecessor and successor, so the unique-predecessor constraint is happy
and the graph is still unresolvable. entity_redirect additionally permits A->B
while B->C, which makes a single-hop read hand back a duplicate.

Neither shape can be expressed as a CHECK constraint, since both need to look at
other rows. They are checked here instead, and the checks are cheap enough to run
as an assertion in tests or after a merge: all five graphs are functional (each
node has at most one outgoing edge, enforced by the unique constraints), and all
five hold exceptional rows rather than bulk data, so loading the edges and
walking them in Python beats a recursive CTE on both clarity and cost.
"""

from __future__ import annotations

from typing import Mapping, TypeVar
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import InstrumentedAttribute

from predictelection.sql.claim import ClaimAssertion, ClaimSupersession
from predictelection.sql.entity import EntityRedirect
from predictelection.sql.polling import PollAverageRevision, PollRevision


_Node = TypeVar("_Node")


def find_cycle(successor: Mapping[_Node, _Node]) -> list[_Node] | None:
    """Return one cycle in a functional graph, or None if it is acyclic.

    The returned list starts and ends on the same node so the cycle reads back as
    a path, e.g. [A, B, A].
    """

    settled: set[_Node] = set()
    for start in successor:
        if start in settled:
            continue
        position: dict[_Node, int] = {}
        path: list[_Node] = []
        node: _Node | None = start
        while node is not None and node not in settled:
            if node in position:
                return [*path[position[node] :], node]
            position[node] = len(path)
            path.append(node)
            node = successor.get(node)
        settled.update(path)
    return None


def _edges(
    session: Session,
    source: InstrumentedAttribute[uuid.UUID],
    target: InstrumentedAttribute[uuid.UUID] | InstrumentedAttribute[uuid.UUID | None],
) -> dict[uuid.UUID, uuid.UUID]:
    return {
        source_id: target_id
        for source_id, target_id in session.execute(
            select(source, target).where(target.is_not(None))
        )
    }


def find_entity_redirect_cycle(session: Session) -> list[uuid.UUID] | None:
    return find_cycle(
        _edges(
            session,
            EntityRedirect.duplicate_entity_id,
            EntityRedirect.canonical_entity_id,
        )
    )


def find_entity_redirect_chains(session: Session) -> list[tuple[uuid.UUID, uuid.UUID]]:
    """Redirects pointing at an entity that is itself redirected.

    Callers should always redirect straight at the terminal canonical entity.
    When they do not, resolve_entity still returns the right answer but every
    naive single-hop join silently reads a duplicate.
    """

    edges = _edges(
        session,
        EntityRedirect.duplicate_entity_id,
        EntityRedirect.canonical_entity_id,
    )
    return [
        (duplicate, canonical)
        for duplicate, canonical in edges.items()
        if canonical in edges
    ]


def find_claim_supersession_cycle(session: Session) -> list[uuid.UUID] | None:
    return find_cycle(
        _edges(
            session,
            ClaimSupersession.predecessor_claim_id,
            ClaimSupersession.successor_claim_id,
        )
    )


def find_claim_assertion_supersession_cycle(session: Session) -> list[uuid.UUID] | None:
    return find_cycle(
        _edges(session, ClaimAssertion.id, ClaimAssertion.supersedes_assertion_id)
    )


def find_poll_revision_supersession_cycle(session: Session) -> list[uuid.UUID] | None:
    return find_cycle(
        _edges(session, PollRevision.id, PollRevision.supersedes_revision_id)
    )


def find_poll_average_revision_supersession_cycle(
    session: Session,
) -> list[uuid.UUID] | None:
    return find_cycle(
        _edges(
            session,
            PollAverageRevision.id,
            PollAverageRevision.supersedes_revision_id,
        )
    )


_CYCLE_CHECKS = (
    ("entity_redirect", find_entity_redirect_cycle),
    ("claim_supersession", find_claim_supersession_cycle),
    ("claim_assertion supersession", find_claim_assertion_supersession_cycle),
    ("poll_revision supersession", find_poll_revision_supersession_cycle),
    (
        "poll_average_revision supersession",
        find_poll_average_revision_supersession_cycle,
    ),
)


def check_graph_integrity(session: Session) -> list[str]:
    """Describe every cycle or non-terminal redirect currently stored."""

    problems: list[str] = []
    for label, check in _CYCLE_CHECKS:
        cycle = check(session)
        if cycle is not None:
            path = " -> ".join(str(node) for node in cycle)
            problems.append(f"{label} cycle: {path}")

    for duplicate, canonical in find_entity_redirect_chains(session):
        problems.append(
            f"entity_redirect {duplicate} points at {canonical}, "
            "which is itself redirected"
        )
    return problems


def assert_graph_integrity(session: Session) -> None:
    """Raise if any supersession or redirect graph is unresolvable."""

    problems = check_graph_integrity(session)
    if problems:
        raise ValueError("; ".join(problems))
