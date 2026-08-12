"""pg_trgm, for fuzzy candidate lookup during pollster resolution

Pollster names arrive from prose — "EPIC-MRA", "EPIC MRA", "Emerson College",
"Emerson College Polling" — and `normalize_entity_name` deliberately folds case
and whitespace only, so near-misses like these do not collapse into one alias
key. Trigram similarity is how resolution *finds* the candidates it should
worry about; it never decides a merge on its own (a wrong merge attributes one
pollster's polls to another silently, a fork is recoverable), it feeds a
ReviewTask.

In-database rather than embeddings because it is deterministic, costs nothing
per call, and needs no service: `similarity()` over a GIN index answers "what
existing names look like this one" in one query.

The index is partial-ish only in spirit — it covers every alias, but the only
reader is pollster resolution, which filters by entity kind after the candidate
fetch. If alias volume ever makes that wasteful, scope it then.

`create_schema` (tests, empty databases) does not run migrations, so the
extension is also ensured there; this revision is what upgrades a live
database, where `create_all` would silently do nothing.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-12

"""

from typing import Sequence, Union

from alembic import op


revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_entity_alias_normalized_trgm "
        "ON entity_alias USING gin (normalized_name gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_entity_alias_normalized_trgm")
    # The extension stays: other consumers may exist by the time anyone
    # downgrades, and dropping it would take their indexes with it.
