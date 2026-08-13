"""Finding an entity a person is looking for, and showing which one it is.

Distinct from `sql.lookup.find_entities`, which exists so an *agent* can avoid
forking an entity and orders alphabetically because a prompt is a list rather
than a ranking. Typed into a search box, that ordering is unusable: searching
`crowley` against this graph returned three towns called Crowley and two FEC
filers before David Crowley, who was not in the top five at all. With 47,039
jurisdictions loaded, alphabetical order means the noise always wins.

## The ranking, and why it is this one

Measured against the real graph rather than reasoned about, because the obvious
answer is wrong. Ranking by `similarity()` puts *Crowley city* (0.67) above
*David Crowley* (0.57): trigram similarity compares whole strings, so every
extra word in a name counts against it, and short names win by being short.

`word_similarity(query, name)` asks a different question — does the query match
some run of whole words inside the name — and answers 1.0 for both. That removes
the length bias and leaves a tie, which is honest: on the name alone they *are*
equally good matches.

The tie-break is how much the graph knows about the entity. A jurisdiction
nobody has asserted anything about is real but is almost never what someone
searching an elections tool meant, and an entity with claims is one somebody has
already researched. Raw similarity breaks any remaining tie, which is where
`UC Berkeley` gets ahead of `Berkeley city`.

Known limitation: **the claim count is blind to polls.** Polls live in their own
tables and are not claims, so a pollster scores zero however much polling it has
done, and loses to a same-named town. Visible in a search for `trafalgar`.
Counting poll revisions too would fix it and costs another join.

## Why the candidate set is filtered before it is ranked

`word_similarity` cannot use the trigram index, so ranking every alias directly
is a sequential scan: **841 ms** over 68,921 aliases, versus **7 ms** filtering
with the indexed `%` operator first. A search box cannot spend 841 ms per
keystroke, so candidates come from the index and ranking happens over the
handful that survive.

The cost is recall. `%` keeps rows above `pg_trgm.similarity_threshold` (0.3 by
default, unchanged here), which is a *whole-string* comparison — so a short
query against a very long name can be filtered out before ranking ever sees it,
even though `word_similarity` would have scored it 1.0. If that starts to matter,
the fix is a GiST index supporting the `<%` operator rather than a change here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import uuid

from sqlalchemy import Row, Select, func, select
from sqlalchemy.orm import Session

from predictelection.query.claims import EntityRef
from predictelection.sql.claim import Claim
from predictelection.sql.entity import (
    Entity,
    EntityAlias,
    EntityIdentifier,
    EntityKind,
    normalize_entity_name,
    resolve_entity,
)


DEFAULT_LIMIT = 20

MATCH_FLOOR = 0.5
"""How much of the query has to appear as words in a name to be offered.

Applied to `word_similarity` after the index has narrowed the field, so the
threshold is this module's and not whatever `pg_trgm.similarity_threshold`
happens to be set to in a given database.
"""

_CANDIDATE_LIMIT = 200
"""How many index hits to rank. Large enough that the best answer is in there,
small enough that ranking is free."""


@dataclass(frozen=True, slots=True)
class EntityHit:
    """One search result, with enough to tell it from its namesakes."""

    entity: EntityRef
    score: float
    context: str | None = None
    """The disambiguator: an OCD id for a jurisdiction, a party for a person, the
    race coordinates for a contest. Without it a result list containing
    `Crowley city` twice is unusable — and this graph contains exactly that."""

    matched_alias: str | None = None
    """Which spelling matched, when it was not the canonical name. A hit on an
    alias with no visible reason looks like a bug."""


def search_entities(
    session: Session,
    query: str,
    *,
    kind: EntityKind | None = None,
    limit: int = DEFAULT_LIMIT,
) -> tuple[EntityHit, ...]:
    """Entities matching a typed query, best first."""

    cleaned = query.strip()
    if not cleaned:
        return ()
    normalized = normalize_entity_name(cleaned)

    word_score = func.word_similarity(normalized, EntityAlias.normalized_name)
    whole_score = func.similarity(EntityAlias.normalized_name, normalized)

    candidates = (
        select(
            EntityAlias.entity_id,
            EntityAlias.name.label("alias"),
            word_score.label("word_score"),
            whole_score.label("whole_score"),
        )
        # The indexed operator, purely to narrow. Ranking happens below.
        .where(EntityAlias.normalized_name.op("%")(normalized))
        .where(word_score >= MATCH_FLOOR)
        .order_by(word_score.desc(), whole_score.desc())
        .limit(_CANDIDATE_LIMIT)
        .subquery()
    )

    statement: Select = (
        select(
            Entity,
            candidates.c.alias,
            func.max(candidates.c.word_score),
            func.max(candidates.c.whole_score),
            func.count(Claim.id.distinct()),
        )
        .join(candidates, candidates.c.entity_id == Entity.id)
        .outerjoin(
            Claim,
            (Claim.subject_id == Entity.id) | (Claim.object_id == Entity.id),
        )
        .group_by(Entity.id, candidates.c.alias)
    )
    if kind is not None:
        statement = statement.where(Entity.kind == kind)

    rows = session.execute(
        statement.order_by(
            func.max(candidates.c.word_score).desc(),
            func.count(Claim.id.distinct()).desc(),
            func.max(candidates.c.whole_score).desc(),
            Entity.canonical_name,
        )
        # Over-fetch, because merged duplicates collapse below and would
        # otherwise return a short page.
        .limit(limit * 2)
    ).all()

    return _resolved(session, rows, limit=limit)


def _resolved(
    session: Session, rows: Sequence[Row[Any]], *, limit: int
) -> tuple[EntityHit, ...]:
    """Collapse merged entities, keeping the best-ranked hit for each survivor.

    A merge is a read-time redirect rather than a rewrite, so the alias that
    matched still belongs to the entity a reviewer merged away — and searching
    `berkeley` returned `UC Berkeley` twice, once as itself and once as the
    duplicate folded into `UC Berkeley IGS`. Offering a user a link to an entity
    the graph considers dead is the same bug that made `resolve_pollster` undo
    its own merges, one layer up.

    Rows arrive ranked, so the first hit for a survivor is its best one.
    """

    best: dict[uuid.UUID, tuple[Entity, str, float]] = {}
    for entity_row, alias, word_score_value, _, _ in rows:
        canonical_id = resolve_entity(session, entity_row.id)
        if canonical_id in best:
            continue
        survivor = (
            entity_row
            if canonical_id == entity_row.id
            else session.get(Entity, canonical_id)
        )
        if survivor is None:  # pragma: no cover - a redirect to nothing
            continue
        best[canonical_id] = (survivor, alias, float(word_score_value))
        if len(best) >= limit:
            break

    contexts = _contexts(session, [found for found, _, _ in best.values()])
    return tuple(
        EntityHit(
            entity=EntityRef(
                entity_id=found.id, kind=found.kind, name=found.canonical_name
            ),
            score=score,
            context=contexts.get(found.id),
            matched_alias=alias if alias != found.canonical_name else None,
        )
        for found, alias, score in best.values()
    )


def entity(session: Session, entity_id: uuid.UUID) -> EntityRef | None:
    """One entity as a reference, following any merge.

    Through `resolve_entity` so a link to an entity a reviewer has since merged
    away lands on the survivor rather than 404ing or showing a dead duplicate.
    """

    found = session.get(Entity, resolve_entity(session, entity_id))
    if found is None:
        return None
    return EntityRef(entity_id=found.id, kind=found.kind, name=found.canonical_name)


def entity_detail(session: Session, entity_id: uuid.UUID) -> EntityHit | None:
    """One entity with its disambiguating context, for a detail page header."""

    reference = entity(session, entity_id)
    if reference is None:
        return None
    found = session.get(Entity, reference.entity_id)
    assert found is not None
    return EntityHit(
        entity=reference,
        score=1.0,
        context=_contexts(session, [found]).get(found.id),
    )


# --------------------------------------------------------------------------


def _contexts(session: Session, entities: Sequence[Entity]) -> dict[uuid.UUID, str]:
    """A one-line disambiguator per entity, in two batched queries.

    Identifiers first because they are the cheapest thing that actually
    distinguishes namesakes: an OCD id names the state a `Crowley city` is in,
    and a contest key carries the whole race. Party is a claim lookup and worth
    the second query — for people it is the only context that means anything.
    """

    if not entities:
        return {}

    ids = [found.id for found in entities]
    context: dict[uuid.UUID, str] = {}

    identifiers = session.execute(
        select(
            EntityIdentifier.entity_id,
            EntityIdentifier.namespace,
            EntityIdentifier.value,
        )
        .where(EntityIdentifier.entity_id.in_(ids))
        .order_by(EntityIdentifier.namespace)
    ).all()
    for entity_id, namespace, value in identifiers:
        context.setdefault(entity_id, f"{namespace}:{value}")

    people = [found.id for found in entities if found.kind is EntityKind.PERSON]
    if people:
        from predictelection.sql.predicate import get_predicate_spec

        parties = session.execute(
            select(Claim.subject_id, Entity.canonical_name)
            .join(Entity, Entity.id == Claim.object_id)
            .where(
                Claim.subject_id.in_(people),
                Claim.predicate_version_id
                == get_predicate_spec("party_affiliation").predicate_version_id,
            )
        ).all()
        for entity_id, party in parties:
            # Party wins over an FEC id for a person: "Democratic" tells a
            # reader which David Crowley this is; "fec:H0WI00123" does not.
            context[entity_id] = party

    return context


__all__ = [
    "DEFAULT_LIMIT",
    "MATCH_FLOOR",
    "EntityHit",
    "entity",
    "entity_detail",
    "search_entities",
]
