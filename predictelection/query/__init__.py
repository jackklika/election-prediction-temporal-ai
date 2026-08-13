"""Asking the graph questions, for a human or an API rather than for an agent.

`sql.lookup` exists so an *agent* can avoid forking an entity: it answers "is
this thing already here, and what did you call it". This package answers the
questions the README's third target is about — who is running, how did polling
move, who won, what is still unreviewed.

Four rules hold it together:

**Claims are the general case; everything else is an exception with a reason.**
`claims_about` and `claims_with` cover any predicate, which is the payoff of an
ontology that stores donations, endorsements and candidacies in the same shape.
A projection earns its place here only by needing something claims cannot give:
`poll_timeline` reads the polling tables because polls are not claims, and
`candidates_in` exists because "who was running on a date" is an interval query
people get wrong by hand.

**Nothing here writes.** These functions are safe to call from a request handler,
in any order, concurrently. The review functions that *do* write stay in
`sql.review_queue`, so an API can grant read access by importing only this.

**Review is respected, not ignored.** A poll can hold two revisions with
different numbers, and which one to believe is a `ReviewDecision`, not a column.
"The latest revision" is the wrong default and this package never uses it.

**Nothing is described from a field it does not have.** `contest_detail` reads
the contest key rather than claims because the structural claims do not exist
yet; `ClaimRow.value` is parsed into the model its predicate declares rather
than handed over as an untyped dict.

Returns frozen dataclasses, not ORM rows: a detached ORM object lazy-loads on
attribute access and raises outside its session, which is exactly what a
serializer does to it. These carry what the caller asked for and nothing else.
"""

from __future__ import annotations

from predictelection.query.claims import (
    ClaimRow,
    EntityRef,
    Evidence,
    claims_about,
    claims_with,
    evidence_for,
)
from predictelection.query.contests import (
    CandidateStint,
    ContestDetail,
    ContestResultRow,
    candidates_in,
    contest_by_key,
    contest_detail,
    contests_in,
    results_for,
    winners_by_office,
)
from predictelection.query.entities import (
    EntityHit,
    entity,
    entity_detail,
    search_entities,
)
from predictelection.query.polls import (
    PollPoint,
    PollReadingRow,
    poll_timeline,
)
from predictelection.query.review import (
    ReviewBacklog,
    unreviewed,
)


__all__ = [
    "CandidateStint",
    "ClaimRow",
    "ContestDetail",
    "ContestResultRow",
    "EntityHit",
    "EntityRef",
    "Evidence",
    "PollPoint",
    "PollReadingRow",
    "ReviewBacklog",
    "candidates_in",
    "claims_about",
    "claims_with",
    "contest_by_key",
    "contest_detail",
    "contests_in",
    "entity",
    "entity_detail",
    "evidence_for",
    "poll_timeline",
    "results_for",
    "search_entities",
    "unreviewed",
    "winners_by_office",
]
