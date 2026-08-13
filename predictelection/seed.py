"""Bring a live database's catalog rows up to what the code declares.

    make seed

Predicates and identifier namespaces are *data*, not schema: `PREDICATE_SPECS`
and `NAMESPACE_SPECS` are the source of truth and `seed_predicates` /
`seed_identifier_namespaces` insert what is missing. That is a deliberately good
property — adding a predicate needs no migration, no enum change and no CHECK
edit, which is what makes a new domain cheap.

It has one consequence that used to have no answer. Seeding ran only from
`create_schema`, which also creates tables, so it happened on a fresh database
and in the test suite and nowhere else. On a database maintained by Alembic — the
only kind with real data in it — a newly declared predicate reached the code but
never the database, and the first claim written against it failed on a foreign
key to a `predicate_version` row that was never inserted.

So this is the missing half of `make migrate`: migrations move the schema,
this moves the catalog. Both are idempotent and both are safe to re-run.

Deliberately not folded into `make migrate`: a migration is a schema change with
a revision history, and seeding is not. Running them together would make it
tempting to write a migration that seeds, which is how catalog rows end up
duplicated across a revision and a spec that later disagree.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from predictelection.clients.sqlalchemy_engine import SqlAlchemyEngineClient
from predictelection.sql import (
    NAMESPACE_SPECS,
    PREDICATE_SPECS,
    PredicateSpec,
    PredicateVersion,
    seed_identifier_namespaces,
    seed_predicates,
)


def seed(session: Session) -> tuple[PredicateSpec, ...]:
    """Insert missing catalog rows. Returns the predicates that were *new here*.

    Counted before the write rather than taken from `seed_predicates`, which
    returns every version it reconciled rather than only the ones it inserted —
    reporting that number would say "19 new" on a database that already had all
    19, which is exactly the kind of output that reads as work being done.

    `seed_predicates` raises rather than rewriting a version whose schema hash
    changed: editing a spec in place would silently reinterpret every claim
    already written against it. Bump the spec's `version` instead; the old
    contract stays readable and old claims keep pointing at it.
    """

    known = set(session.scalars(select(PredicateVersion.id)))
    added = tuple(
        spec for spec in PREDICATE_SPECS if spec.predicate_version_id not in known
    )

    seed_identifier_namespaces(session)
    seed_predicates(session)
    return added


def main() -> None:
    session_factory = SqlAlchemyEngineClient().session_factory
    with session_factory() as session, session.begin():
        added = seed(session)

    print(f"identifier namespaces  {len(NAMESPACE_SPECS)} declared, reconciled")
    print(f"predicate versions     {len(PREDICATE_SPECS)} declared, reconciled")
    if added:
        print(f"  new here: {', '.join(spec.slug for spec in added)}")


if __name__ == "__main__":
    main()
