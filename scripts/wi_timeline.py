"""The Wisconsin timeline, entirely from stored claims.

The acceptance check for the candidacy lifecycle: who was running on five dates,
the endorsement arc, and the result with `won`. Three different candidate sets
across the dates is the property that proves withdrawals were recorded as
intervals ending rather than rows changing.

    uv run python scripts/wi_timeline.py

Also the standing check on `predictelection.query`. This script used to be a
hundred lines of hand-written SQL — four separate joins from claim to entity to
identifier, each rebuilt slightly differently — which is exactly the duplication
the query module exists to end. It now contains no SQL at all. If a future
question here cannot be asked through `query`, that is the module's gap to close
rather than this file's to work around, and porting this back to raw SQL would
hide the very thing it is meant to reveal.

Writing it did close two: nothing could reach a claim's citation, and the
projections dropped the claim id, so a result row could not be traced back to
what said so. `query.evidence_for` and the `claim_id` fields are both here
because this port needed them.
"""

from datetime import UTC, datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictelection import query
from predictelection.clients.sqlalchemy_engine import SqlAlchemyEngineClient

CONTEST = "ocd-division/country:us/state:wi/governor/2026/primary/democratic"

DATES = [
    ("2026-05-01  (early)", "2026-05-01"),
    ("2026-06-15  (before Crowley withdrew)", "2026-06-15"),
    ("2026-07-10  (Crowley out, backing Rodriguez)", "2026-07-10"),
    ("2026-07-20  (Rodriguez out too)", "2026-07-20"),
    ("2026-08-11  (primary day)", "2026-08-11"),
]

RULE = "=" * 74


def heading(text: str) -> None:
    print(RULE)
    print(text)
    print(RULE)


def day(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


with SqlAlchemyEngineClient().session_factory() as session:
    contest_id = query.contest_by_key(session, CONTEST)
    if contest_id is None:
        raise SystemExit(f"no contest recorded for {CONTEST}")

    heading("WHO WAS RUNNING — from validity intervals alone, no status column")
    for label, value in DATES:
        running = sorted(
            stint.person.name
            for stint in query.candidates_in(session, contest_id, at=day(value))
        )
        print(f"  {label:46} {', '.join(running) or '(nobody)'}")

    print()
    heading("CANDIDACY STINTS — a re-entry is two claims, not an overwrite")
    stints = sorted(
        query.candidates_in(session, contest_id),
        key=lambda stint: (stint.person.name, stint.started_at or datetime.min),
    )
    for stint in stints:
        started = stint.started_at.date() if stint.started_at else "?"
        ended = stint.ended_at.date() if stint.ended_at else "still running"
        precision = f"[{stint.started_precision or '-'}/{stint.ended_precision or '-'}]"
        print(f"  {stint.person.name:24} {started} → {str(ended):14} {precision}")

    print()
    heading("ENDORSEMENTS — switches and withdrawals as separate intervals")
    # Ascending: `claims_with` returns newest first, which is right for a list
    # and wrong for an arc.
    endorsements = sorted(
        query.claims_with(session, "endorsed"),
        key=lambda row: (row.valid_from or datetime.min, row.subject.name),
    )
    for row in endorsements:
        strength = (row.value or {}).get("strength", "?")
        started = row.valid_from.date() if row.valid_from else "?"
        ended = row.valid_to.date() if row.valid_to else "open"
        backed = row.object.name if row.object else "(unnamed)"
        print(
            f"  {row.subject.name:22} → {backed:20} {strength:10} {started} → {ended}"
        )
    if not endorsements:
        print("  (none recorded)")

    print()
    heading("RESULT — votes from the table, `won` from the Nominee heading")
    results = query.results_for(session, contest_id)
    citations = query.evidence_for(session, [result.claim_id for result in results])
    for result in results:
        won = {True: "WON", False: "", None: "(not stated)"}[result.won]
        excerpt = next(
            (
                cited.excerpt
                for cited in citations.get(result.claim_id, ())
                if cited.excerpt
            ),
            "(no excerpt)",
        )
        votes = f"{result.votes:,}" if result.votes is not None else "?"
        share = f"{result.share}" if result.share is not None else "?"
        print(
            f"  {result.candidate.name:24} {votes:>8} {share:>6}%  {won:12} "
            f"page said: {excerpt[:34]}"
        )

    print()
    heading("REVIEW — how much of the above anyone has checked")
    backlog = query.unreviewed(session)
    print(f"  open tasks       {backlog.open}")
    print(
        f"  decided          {backlog.decisions}"
        f"  ({backlog.accepted} accepted, {backlog.rejected} rejected)"
    )
