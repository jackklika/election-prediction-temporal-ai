"""The Wisconsin timeline, entirely from stored claims.

The acceptance check for the candidacy lifecycle: who was running on five dates,
the endorsement arc, and the result with `won`. Three different candidate sets
across the dates is the property that proves withdrawals were recorded as
intervals ending rather than rows changing.

    uv run python scripts/wi_timeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlalchemy as sa
from predictelection.clients.sqlalchemy_engine import PostgresConfig
from predictelection.sql import get_predicate_spec

CONTEST = "ocd-division/country:us/state:wi/governor/2026/primary/democratic"
CAND = get_predicate_spec("candidate_in").predicate_version_id
RESULT = get_predicate_spec("contest_result").predicate_version_id
ENDORSED = get_predicate_spec("endorsed").predicate_version_id

e = sa.create_engine(PostgresConfig().url)  # ty: ignore[missing-argument]
with e.connect() as c:
    print("=" * 74)
    print("WHO WAS RUNNING — from validity intervals alone, no status column")
    print("=" * 74)
    for label, date in [
        ("2026-05-01  (early)", "2026-05-01"),
        ("2026-06-15  (before Crowley withdrew)", "2026-06-15"),
        ("2026-07-10  (Crowley out, backing Rodriguez)", "2026-07-10"),
        ("2026-07-20  (Rodriguez out too)", "2026-07-20"),
        ("2026-08-11  (primary day)", "2026-08-11"),
    ]:
        names = [
            r[0]
            for r in c.execute(
                sa.text("""
            select distinct e.canonical_name
            from claim cl
            join entity e on e.id = cl.subject_id
            join entity_identifier ci on ci.entity_id = cl.object_id
            where cl.predicate_version_id = :v and ci.value = :k
              and cl.valid_from <= :d
              and (cl.valid_to is null or cl.valid_to > :d)
            order by 1"""),
                {"v": CAND, "k": CONTEST, "d": date},
            )
        ]
        print(f"  {label:46} {', '.join(names) or '(nobody)'}")

    print()
    print("=" * 74)
    print("CANDIDACY STINTS — a re-entry is two claims, not an overwrite")
    print("=" * 74)
    for r in c.execute(
        sa.text("""
        select e.canonical_name, cl.valid_from::date, cl.valid_to::date,
               cl.valid_from_precision, cl.valid_to_precision
        from claim cl
        join entity e on e.id = cl.subject_id
        join entity_identifier ci on ci.entity_id = cl.object_id
        where cl.predicate_version_id = :v and ci.value = :k
        order by e.canonical_name, cl.valid_from"""),
        {"v": CAND, "k": CONTEST},
    ):
        end = r[2] or "still running"
        prec = f"[{r[3]}/{r[4] or '-'}]"
        print(f"  {r[0]:24} {r[1]} → {str(end):14} {prec}")

    print()
    print("=" * 74)
    print("ENDORSEMENTS — switches and withdrawals as separate intervals")
    print("=" * 74)
    rows = list(
        c.execute(
            sa.text("""
        select er.canonical_name, ee.canonical_name,
               cl.value->>'strength', cl.valid_from::date, cl.valid_to::date
        from claim cl
        join entity er on er.id = cl.subject_id
        join entity ee on ee.id = cl.object_id
        where cl.predicate_version_id = :v
        order by cl.valid_from, er.canonical_name"""),
            {"v": ENDORSED},
        )
    )
    for r in rows:
        print(f"  {r[0]:22} → {r[1]:20} {r[2]:10} {r[3]} → {r[4] or 'open'}")
    if not rows:
        print("  (none recorded)")

    print()
    print("=" * 74)
    print("RESULT — votes from the table, `won` from the Nominee heading")
    print("=" * 74)
    for r in c.execute(
        sa.text("""
        select e.canonical_name, (cl.value->>'votes')::int, cl.value->>'share',
               cl.value->>'won', ea.excerpt
        from claim cl
        join entity e on e.id = cl.subject_id
        join entity_identifier ci on ci.entity_id = cl.object_id
        join claim_assertion ca on ca.claim_id = cl.id
        join evidence_anchor ea on ea.id = ca.evidence_anchor_id
        where cl.predicate_version_id = :v and ci.value = :k
        order by 2 desc"""),
        {"v": RESULT, "k": CONTEST},
    ):
        won = {"true": "WON", "false": "", None: "(not stated)"}.get(r[3], "")
        print(f"  {r[0]:24} {r[1]:>8,} {r[2]:>6}%  {won:12} page said: {r[4][:34]}")
