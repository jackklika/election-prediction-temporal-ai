"""Work the review queue from the command line.

    make review                        # what is waiting
    make review-next                   # answer them one at a time

Four ingestion paths file review tasks and, until this, nothing answered them.
The queue is where the project's judgement calls were parked: a pollster name
that resembles one already known, two sources printing different numbers for one
poll, a poll with no fieldwork end and therefore no cross-source identity. Every
one of those was deliberately *not* decided automatically, because guessing wrong
is silent and a queue entry is not.

A CLI rather than a UI because the decisions are already recorded append-only in
the database — `ReviewDecision` for the verdict, `EntityRedirect` for a merge —
so a front end would add a rendering layer to a model that is complete without
one. What was missing was a reader.

Three properties worth keeping if this grows:

- **One transaction per decision.** Quitting halfway through the queue keeps
  every answer given so far. A single transaction around the loop would discard
  them all on a stray Ctrl-C.
- **A merge is applied, not requested.** Accepting "these two organizations are
  one" writes the redirect there and then, in the same transaction as the
  decision, so the queue cannot end up saying a merge happened that did not.
- **Nothing here decides anything itself.** No similarity threshold, no
  auto-accept. The reviewer is the point.
"""

from __future__ import annotations

import argparse
import os
import uuid

from sqlalchemy.orm import Session, sessionmaker

from predictelection.clients.sqlalchemy_engine import SqlAlchemyEngineClient
from predictelection.sql import (
    Entity,
    EntityKind,
    ReviewOutcome,
    ReviewTask,
    ReviewTaskStatus,
    ReviewTaskView,
    decide,
    find_entities,
    find_task,
    merge_entities,
    pending_tasks,
    task_view,
)


RULE = "─" * 78


def main() -> None:
    parser = _parser()
    args = parser.parse_args()

    session_factory = SqlAlchemyEngineClient().session_factory
    try:
        if args.command == "list":
            _list(session_factory, status=args.status, limit=args.limit)
        elif args.command == "show":
            _show(session_factory, args.task)
        elif args.command == "next":
            _walk(session_factory, reviewer=_reviewer(args), limit=args.limit)
        elif args.command == "accept":
            _decide_one(
                session_factory,
                args.task,
                outcome=ReviewOutcome.ACCEPTED,
                reviewer=_reviewer(args),
                reason=args.reason,
            )
        elif args.command == "reject":
            _decide_one(
                session_factory,
                args.task,
                outcome=ReviewOutcome.REJECTED,
                reviewer=_reviewer(args),
                reason=args.reason,
            )
        elif args.command == "merge":
            _merge_one(
                session_factory,
                args.task,
                into=args.into,
                reviewer=_reviewer(args),
                reason=args.reason,
                reverse=args.reverse,
            )
        else:  # pragma: no cover - argparse rejects anything else
            parser.error(f"unknown command {args.command}")
    except ValueError as error:
        # A bad prefix, an impossible merge, a missing reason: the reviewer's
        # problem to fix, not a stack trace.
        raise SystemExit(f"error: {error}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="review", description=__doc__)
    parser.add_argument(
        "--reviewer",
        default=None,
        help="Who is deciding. Defaults to $REVIEW_REVIEWER, then $USER.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="Show the queue.")
    listing.add_argument(
        "--status",
        default=ReviewTaskStatus.PENDING.value,
        choices=[*(status.value for status in ReviewTaskStatus), "all"],
    )
    listing.add_argument("--limit", type=int, default=50)

    showing = sub.add_parser("show", help="Show one task in full.")
    showing.add_argument("task", help="Task id, or a unique prefix of one.")

    walking = sub.add_parser("next", help="Answer pending tasks one at a time.")
    walking.add_argument("--limit", type=int, default=50)

    accepting = sub.add_parser("accept", help="The data is right as recorded.")
    accepting.add_argument("task")
    accepting.add_argument("--reason", default=None)

    rejecting = sub.add_parser("reject", help="The data is wrong. Says why.")
    rejecting.add_argument("task")
    rejecting.add_argument("--reason", required=True)

    merging = sub.add_parser("merge", help="Two entities are one. Writes a redirect.")
    merging.add_argument("task")
    merging.add_argument(
        "--into",
        required=True,
        help="Entity to keep: an id, a prefix of one, or a candidate number.",
    )
    merging.add_argument(
        "--reverse",
        action="store_true",
        help="Keep the task's own entity and merge --into it instead.",
    )
    merging.add_argument("--reason", default=None)
    return parser


# ---------------------------------------------------------------- reading


def _list(session_factory: sessionmaker[Session], *, status: str, limit: int) -> None:
    wanted = None if status == "all" else ReviewTaskStatus(status)
    with session_factory() as session:
        views = pending_tasks(session, status=wanted, limit=limit)

    if not views:
        print(f"nothing {status}")
        return

    print(f"{'id':<10}{'pri':<5}{'target':<18}reason")
    print(RULE)
    for view in views:
        reason = (view.reason or view.target.headline).replace("\n", " ")
        print(
            f"{view.short_id:<10}{view.priority:<5}{view.target.kind:<18}"
            f"{_clip(reason, 44)}"
        )
    print(RULE)
    print(f"{len(views)} task(s). `review show <id>` for one of them.")


def _show(session_factory: sessionmaker[Session], prefix: str) -> None:
    with session_factory() as session:
        print(_render(task_view(session, find_task(session, prefix))))


def _render(view: ReviewTaskView) -> str:
    lines = [
        RULE,
        f"{view.short_id}  {view.target.kind}  priority {view.priority}"
        f"  {view.status.value}",
        RULE,
        f"  {view.target.headline}",
        "",
    ]
    width = max((len(label) for label, _ in view.target.details), default=0)
    for label, value in view.target.details:
        lines.append(f"  {label:<{width}}  {value}")
    if view.reason:
        lines += ["", "  flagged because:", f"    {view.reason}"]
    if view.target.candidates:
        # Named for what they are rather than for what a reviewer might do with
        # them: these are organizations resembling this one, and that is true
        # whether the task was filed about the resemblance or about the numbers.
        lines += ["", "  possible duplicates of this pollster:"]
        lines += [
            f"    [{index}] {candidate.canonical_name}"
            f"  (matched on {candidate.name!r})  [{str(candidate.entity_id)[:8]}]"
            for index, candidate in enumerate(view.target.candidates, start=1)
        ]
    return "\n".join(lines)


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else f"{text[: width - 1]}…"


# ---------------------------------------------------------------- deciding


def _decide_one(
    session_factory: sessionmaker[Session],
    prefix: str,
    *,
    outcome: ReviewOutcome,
    reviewer: str,
    reason: str | None,
) -> None:
    with session_factory() as session, session.begin():
        task = find_task(session, prefix)
        decide(
            session,
            task,
            outcome=outcome,
            reviewer=reviewer,
            reason=reason,
            action_token=str(uuid.uuid4()),
        )
        print(f"{str(task.id)[:8]}  {outcome.value}")


def _merge_one(
    session_factory: sessionmaker[Session],
    prefix: str,
    *,
    into: str,
    reviewer: str,
    reason: str | None,
    reverse: bool,
) -> None:
    with session_factory() as session, session.begin():
        task = find_task(session, prefix)
        view = task_view(session, task)
        subject = view.target.subject_entity_id
        if subject is None:
            raise ValueError(f"task {view.short_id} is not about an entity")
        chosen = _resolve_choice(session, view, into)
        duplicate, canonical = (chosen, subject) if reverse else (subject, chosen)
        _apply_merge(
            session,
            task,
            duplicate=duplicate,
            canonical=canonical,
            reviewer=reviewer,
            reason=reason,
        )


def _apply_merge(
    session: Session,
    task: ReviewTask,
    *,
    duplicate: uuid.UUID,
    canonical: uuid.UUID,
    reviewer: str,
    reason: str | None,
) -> None:
    """Redirect and decision together, in the caller's transaction.

    Recorded as ACCEPTED rather than as a correction: once the redirect exists,
    the revision that raised the task points — through the redirect — at the
    right organization, so nothing about the poll itself was wrong. The reason is
    what carries the merge, and it names both entities so the decision is
    readable without joining back to `entity_redirect`.
    """

    explanation = reason or (
        f"{_label(session, duplicate)} and {_label(session, canonical)} "
        "are the same organization"
    )
    merge_entities(
        session,
        duplicate_id=duplicate,
        canonical_id=canonical,
        reviewer=reviewer,
        reason=explanation,
    )
    decide(
        session,
        task,
        outcome=ReviewOutcome.ACCEPTED,
        reviewer=reviewer,
        reason=(
            f"merged {_label(session, duplicate)} into "
            f"{_label(session, canonical)}: {explanation}"
        ),
        action_token=str(uuid.uuid4()),
    )
    print(
        f"{str(task.id)[:8]}  merged {_label(session, duplicate)} "
        f"into {_label(session, canonical)}"
    )


def _label(session: Session, entity_id: uuid.UUID) -> str:
    entity = session.get(Entity, entity_id)
    return f"{entity.canonical_name!r}" if entity else str(entity_id)


def _resolve_choice(session: Session, view: ReviewTaskView, choice: str) -> uuid.UUID:
    """A candidate number, an entity id, a prefix of one, or a name.

    Numbers first because that is what the walk-through prints, and a bare digit
    is never a UUID prefix worth guessing at.
    """

    cleaned = choice.strip()
    candidates = view.target.candidates
    if cleaned.isdigit():
        index = int(cleaned)
        if not 1 <= index <= len(candidates):
            raise ValueError(f"pick 1-{len(candidates)}, not {index}")
        return candidates[index - 1].entity_id

    matches = [
        candidate.entity_id
        for candidate in candidates
        if str(candidate.entity_id).startswith(cleaned.lower())
    ]
    if len(matches) == 1:
        return matches[0]

    try:
        entity_id = uuid.UUID(cleaned)
    except ValueError:
        return _by_name(session, cleaned)
    if session.get(Entity, entity_id) is None:
        raise ValueError(f"no entity {entity_id}")
    return entity_id


def _by_name(session: Session, name: str) -> uuid.UUID:
    """Last resort: name the organization instead of its id.

    Reuses `find_entities`, so a name typed here matches the same way the agents'
    lookup does — canonical name substring or a known alias.
    """

    found = find_entities(session, name=name, kind=EntityKind.ORGANIZATION, limit=5)
    if not found:
        raise ValueError(f"no organization matches {name!r}")
    if len(found) > 1:
        listed = ", ".join(f"{match.canonical_name!r}" for match in found)
        raise ValueError(f"{name!r} matches several organizations: {listed}")
    return found[0].entity_id


# ---------------------------------------------------------------- the walk


def _walk(session_factory: sessionmaker[Session], *, reviewer: str, limit: int) -> None:
    """The queue one task at a time, a session per decision.

    Re-reads the pending list once at the start and then works by id: a merge
    changes what later tasks should show — two of them can name the same pair of
    organizations — so each task's view is built fresh when it comes up.
    """

    with session_factory() as session:
        task_ids = [view.task_id for view in pending_tasks(session, limit=limit)]

    if not task_ids:
        print("nothing pending")
        return

    print(f"{len(task_ids)} pending. a=accept r=reject m=merge s=skip q=quit\n")
    answered = 0
    for position, task_id in enumerate(task_ids, start=1):
        with session_factory() as session, session.begin():
            task = session.get(ReviewTask, task_id)
            if task is None or task.status is not ReviewTaskStatus.PENDING:
                continue  # answered by an earlier action, or by someone else
            view = task_view(session, task)
            print(f"\n({position}/{len(task_ids)})")
            print(_render(view))

            action = _prompt("\n  [a]ccept [r]eject [m]erge [s]kip [q]uit > ").lower()
            if action.startswith("q"):
                print(f"\n{answered} answered, {len(task_ids) - position + 1} left")
                return
            if action.startswith("s") or not action:
                continue
            if action.startswith("a"):
                decide(
                    session,
                    task,
                    outcome=ReviewOutcome.ACCEPTED,
                    reviewer=reviewer,
                    reason=_prompt("  note (optional) > ") or None,
                    action_token=str(uuid.uuid4()),
                )
                print("  accepted")
            elif action.startswith("r"):
                reason = _prompt("  why is it wrong? > ")
                if not reason:
                    print("  a rejection needs a reason; skipped")
                    continue
                decide(
                    session,
                    task,
                    outcome=ReviewOutcome.REJECTED,
                    reviewer=reviewer,
                    reason=reason,
                    action_token=str(uuid.uuid4()),
                )
                print("  rejected")
            elif action.startswith("m"):
                if not _merge_interactively(session, task, view, reviewer=reviewer):
                    continue
            else:
                print("  did not understand that; skipped")
                continue
            answered += 1

    print(f"\n{answered} answered")


def _merge_interactively(
    session: Session, task: ReviewTask, view: ReviewTaskView, *, reviewer: str
) -> bool:
    """Pick a candidate and a direction. False means nothing was written."""

    subject = view.target.subject_entity_id
    if subject is None or not view.target.candidates:
        print("  nothing to merge with here; skipped")
        return False

    choice = _prompt(f"  merge with which? [1-{len(view.target.candidates)}] > ")
    if not choice:
        return False
    try:
        chosen = _resolve_choice(session, view, choice)
    except ValueError as error:
        print(f"  {error}; skipped")
        return False

    keep = _prompt(
        f"  which name survives? [1] {_label(session, subject)} "
        f"[2] {_label(session, chosen)} > "
    )
    if keep not in {"1", "2"}:
        print("  no direction chosen; skipped")
        return False

    duplicate, canonical = (chosen, subject) if keep == "1" else (subject, chosen)
    try:
        _apply_merge(
            session,
            task,
            duplicate=duplicate,
            canonical=canonical,
            reviewer=reviewer,
            reason=_prompt("  note (optional) > ") or None,
        )
    except ValueError as error:
        print(f"  {error}; skipped")
        return False
    return True


def _prompt(text: str) -> str:
    try:
        return input(text).strip()
    except EOFError:
        # Piped input that ran out is a quit, not a crash.
        print()
        raise SystemExit(0) from None


def _reviewer(args: argparse.Namespace) -> str:
    """Who to attribute decisions to.

    Refuses to guess: `ck_review_decision_reviewer_identifier_nonempty` would
    reject a blank one anyway, and an anonymous decision is not much use to the
    next person reading the queue.
    """

    who = args.reviewer or os.environ.get("REVIEW_REVIEWER") or os.environ.get("USER")
    if not who or not who.strip():
        raise SystemExit("who is reviewing? pass --reviewer or set $REVIEW_REVIEWER")
    return who.strip()


if __name__ == "__main__":
    main()
