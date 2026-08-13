"""Reading the review queue and answering it.

The queue was written by four ingestion paths and read by nothing, so these are
the first tests of the other half: that a task can be seen, that a verdict is
recorded without mutating what it judged, and — the one that matters most — that
a merge survives the next import rather than being quietly undone by it.
"""

from __future__ import annotations

from datetime import UTC, datetime
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection.sql import (
    Entity,
    EntityKind,
    EntityRedirect,
    ReviewDecision,
    ReviewerKind,
    ReviewOutcome,
    ReviewTask,
    ReviewTaskStatus,
    TimePrecision,
    decide,
    find_entity_redirect_chains,
    find_task,
    merge_entities,
    new_review_decision,
    normalize_entity_name,
    pending_tasks,
    resolve_entity,
    task_view,
)
from predictelection.research.contests import normalize_slug
from predictelection.tests.factories import make_entity, make_poll_revision, new_alias


pytestmark = pytest.mark.postgres


def _task(session: Session, *, reason: str = "worth a look", priority: int = 50):
    pollster = make_entity(
        session, kind=EntityKind.ORGANIZATION, canonical_name="Example Polling"
    )
    revision = make_poll_revision(
        session,
        pollster_id=pollster.id,
        created_by="import_wikipedia_polls",
        fieldwork_started_on=datetime(2026, 3, 1, tzinfo=UTC).date(),
        fieldwork_ended_on=datetime(2026, 3, 4, tzinfo=UTC).date(),
    )
    task = ReviewTask(poll_revision=revision, reason=reason, priority=priority)
    session.add(task)
    session.flush()
    return task


def test_the_queue_comes_back_most_urgent_first(session: Session) -> None:
    """Priority ascending, then age. A queue in insertion order is not one."""

    _task(session, reason="middling", priority=50)
    _task(session, reason="urgent", priority=10)
    _task(session, reason="whenever", priority=90)

    views = pending_tasks(session)

    assert [view.reason for view in views] == ["urgent", "middling", "whenever"]
    assert all(view.status is ReviewTaskStatus.PENDING for view in views)


def test_a_task_carries_enough_of_its_target_to_judge_it(session: Session) -> None:
    """A reason naming two organizations is useless without their ids."""

    task = _task(session)
    view = task_view(session, task)

    assert view.target.kind == "poll_revision"
    assert view.target.subject_entity_id is not None
    assert "Example Polling" in view.target.headline
    labels = dict(view.target.details)
    assert "Example Polling" in labels["pollster"]
    assert labels["fieldwork"]


def test_a_completed_task_is_not_pending(session: Session) -> None:
    task = _task(session)
    decide(
        session,
        task,
        outcome=ReviewOutcome.ACCEPTED,
        reviewer="jack",
        action_token=str(uuid.uuid4()),
    )

    assert task.status is ReviewTaskStatus.COMPLETED
    assert task.completed_at is not None  # ck_completion_matches_status
    assert not pending_tasks(session)


def test_deciding_twice_is_refused_rather_than_silently_reopened(
    session: Session,
) -> None:
    """The queue's uniqueness index only covers open tasks.

    A second decision on a closed task would write a second `ReviewDecision`
    against the same target with nothing to say which is current — the table
    orders by seq, so it would silently become the answer.
    """

    task = _task(session)
    decide(
        session,
        task,
        outcome=ReviewOutcome.ACCEPTED,
        reviewer="jack",
        action_token=str(uuid.uuid4()),
    )

    with pytest.raises(ValueError, match="already completed"):
        decide(
            session,
            task,
            outcome=ReviewOutcome.REJECTED,
            reviewer="jack",
            reason="changed my mind",
            action_token=str(uuid.uuid4()),
        )


def test_a_rejection_without_a_reason_is_refused_at_the_call_site(
    session: Session,
) -> None:
    """ck_review_decision_nonacceptance_has_reason says the same thing on flush.

    Raising here means the reviewer is told while they are still looking at the
    task, rather than by an IntegrityError after the transaction is built.
    """

    task = _task(session)
    with pytest.raises(ValueError, match="must say why"):
        decide(
            session,
            task,
            outcome=ReviewOutcome.REJECTED,
            reviewer="jack",
            reason="   ",
            action_token=str(uuid.uuid4()),
        )
    assert task.status is ReviewTaskStatus.PENDING


def test_a_reviewer_can_change_their_mind_back(session: Session) -> None:
    """Why the decision key is a per-action token and not a content hash.

    `idempotency_key` documents this exception explicitly. Keyed on
    (target, outcome, reason), an accept-reject-accept sequence would collide on
    the third step and turn a deliberate reversal into a silent no-op.
    """

    revision = make_poll_revision(session)
    for outcome, reason in [
        (ReviewOutcome.ACCEPTED, None),
        (ReviewOutcome.REJECTED, "on reflection, wrong"),
        (ReviewOutcome.ACCEPTED, None),
    ]:
        session.add(
            new_review_decision(
                action_token=str(uuid.uuid4()),
                outcome=outcome,
                reviewer_identifier="jack",
                reason=reason,
                poll_revision_id=revision.id,
            )
        )
    session.flush()

    decisions = session.scalars(
        select(ReviewDecision)
        .where(ReviewDecision.poll_revision_id == revision.id)
        .order_by(ReviewDecision.seq)
    ).all()
    assert [decision.outcome for decision in decisions] == [
        ReviewOutcome.ACCEPTED,
        ReviewOutcome.REJECTED,
        ReviewOutcome.ACCEPTED,
    ]


def test_a_decision_must_name_exactly_one_target(session: Session) -> None:
    with pytest.raises(ValueError, match="exactly one target"):
        new_review_decision(
            action_token="t",
            outcome=ReviewOutcome.ACCEPTED,
            reviewer_identifier="jack",
        )


def test_a_decision_must_say_who_made_it(session: Session) -> None:
    revision = make_poll_revision(session)
    with pytest.raises(ValueError, match="who made it"):
        new_review_decision(
            action_token="t",
            outcome=ReviewOutcome.ACCEPTED,
            reviewer_identifier="  ",
            poll_revision_id=revision.id,
        )


def test_an_automated_decision_is_recorded_as_one(session: Session) -> None:
    """The reviewer kind exists so a rule's verdict is not read as a person's."""

    task = _task(session)
    decision = decide(
        session,
        task,
        outcome=ReviewOutcome.ACCEPTED,
        reviewer="dedup-rule",
        reviewer_kind=ReviewerKind.AUTOMATED_RULE,
        action_token=str(uuid.uuid4()),
    )
    assert decision.reviewer_kind is ReviewerKind.AUTOMATED_RULE


# ------------------------------------------------------------------ merging


def _org(session: Session, name: str) -> Entity:
    """An organization with its name recorded as an alias, as ingestion leaves it.

    The alias is not decoration: `pollster_lookalikes` matches on the alias
    index, so an entity without one is invisible to the merge-candidate lookup.
    """

    entity = make_entity(session, kind=EntityKind.ORGANIZATION, canonical_name=name)
    # Keyed by normalized form, because uq_entity_alias_identity is: a one-word
    # name and its own slug are the same alias row.
    spellings = {normalize_entity_name(spelling): spelling for spelling in (name,)}
    spellings.setdefault(
        normalize_entity_name(normalize_slug(name)), normalize_slug(name)
    )
    for spelling in spellings.values():
        session.add(new_alias(entity.id, spelling))
    session.flush()
    return entity


def test_a_merge_makes_the_duplicate_resolve_to_the_survivor(
    session: Session,
) -> None:
    duplicate = _org(session, "UC Berkeley")
    canonical = _org(session, "UC Berkeley IGS")

    merge_entities(
        session,
        duplicate_id=duplicate.id,
        canonical_id=canonical.id,
        reviewer="jack",
        reason="same institute, two spellings",
    )

    assert resolve_entity(session, duplicate.id) == canonical.id
    assert session.scalar(select(func.count(EntityRedirect.duplicate_entity_id))) == 1


def test_a_new_merge_points_at_the_surviving_entity_not_a_duplicate(
    session: Session,
) -> None:
    """Naming an entity that has itself been merged away still lands on the end.

    `find_entity_redirect_chains` calls a redirect-to-a-redirect a defect,
    because every single-hop join then reads a duplicate. The one chain that is
    unavoidable is the one made by history: `EntityRedirect` is immutable, so
    merging B into C cannot repoint an A -> B written before it. What *is*
    avoidable is writing a new one, and this is that guarantee.
    """

    first = _org(session, "Trafalgar Group (R)")
    second = _org(session, "The Trafalgar Group (R)")
    third = _org(session, "Trafalgar Group/InsiderAdvantage")
    latecomer = _org(session, "Trafalgar")

    merge_entities(
        session,
        duplicate_id=first.id,
        canonical_id=second.id,
        reviewer="jack",
        reason="same pollster",
    )
    merge_entities(
        session,
        duplicate_id=second.id,
        canonical_id=third.id,
        reviewer="jack",
        reason="and the same as this one",
    )
    merge_entities(
        session,
        duplicate_id=latecomer.id,
        canonical_id=second.id,  # already a duplicate itself
        reviewer="jack",
        reason="all one organization",
    )

    written = session.get(EntityRedirect, latecomer.id)
    assert written is not None
    assert written.canonical_entity_id == third.id, "resolved past the duplicate"
    assert resolve_entity(session, first.id) == third.id
    assert resolve_entity(session, latecomer.id) == third.id

    # the only chain is the historical one, and it is the pair that made it
    assert find_entity_redirect_chains(session) == [(first.id, second.id)]


def test_a_merge_refuses_the_shapes_that_would_corrupt_the_graph(
    session: Session,
) -> None:
    one = _org(session, "Echelon Insights")
    two = _org(session, "Echleon Insights")

    with pytest.raises(ValueError, match="duplicate of itself"):
        merge_entities(
            session,
            duplicate_id=one.id,
            canonical_id=one.id,
            reviewer="jack",
            reason="typo",
        )
    with pytest.raises(ValueError, match="must say why"):
        merge_entities(
            session,
            duplicate_id=one.id,
            canonical_id=two.id,
            reviewer="jack",
            reason="  ",
        )

    merge_entities(
        session,
        duplicate_id=one.id,
        canonical_id=two.id,
        reviewer="jack",
        reason="misspelling of the same firm",
    )
    with pytest.raises(ValueError, match="already been merged"):
        merge_entities(
            session,
            duplicate_id=one.id,
            canonical_id=_org(session, "Somebody Else").id,
            reviewer="jack",
            reason="second thoughts",
        )
    # and the reverse direction, which would leave neither with a canonical form
    with pytest.raises(ValueError, match="duplicate of itself"):
        merge_entities(
            session,
            duplicate_id=two.id,
            canonical_id=one.id,
            reviewer="jack",
            reason="the other way round",
        )


def test_a_merge_candidate_is_offered_under_its_surviving_name(
    session: Session,
) -> None:
    """Lookalikes are resolved before being offered.

    Otherwise the second reviewer to look at a pair is offered the entity the
    first one already merged away — a merge onto a duplicate, which is how a
    chain gets made.
    """

    task = _task(session)
    resembling = _org(session, "Example Pollings")
    survivor = _org(session, "Example Polling Co")
    merge_entities(
        session,
        duplicate_id=resembling.id,
        canonical_id=survivor.id,
        reviewer="jack",
        reason="already handled",
    )

    view = task_view(session, task)
    offered = {candidate.entity_id for candidate in view.target.candidates}
    assert resembling.id not in offered
    assert survivor.id in offered


# ------------------------------------------------------------------ lookup


def test_a_task_is_addressable_by_a_prefix(session: Session) -> None:
    """A queue that can only be addressed by full UUID is not one a person works."""

    task = _task(session)
    assert find_task(session, str(task.id)[:8]).id == task.id
    assert find_task(session, str(task.id)).id == task.id

    with pytest.raises(ValueError, match="no review task"):
        find_task(session, "ffffffffff")
    with pytest.raises(ValueError, match="give a task id"):
        find_task(session, "  ")


def test_an_ambiguous_prefix_refuses_rather_than_picking(session: Session) -> None:
    """Two tasks, one prefix: guessing would decide the wrong one."""

    first = _task(session)
    second = _task(session)
    second.id = uuid.UUID(f"{str(first.id)[:8]}{str(second.id)[8:]}")
    session.flush()

    with pytest.raises(ValueError, match="matches 2 tasks"):
        find_task(session, str(first.id)[:8])


def test_a_claim_task_renders_as_the_sentence_it_asserts(session: Session) -> None:
    """Claims are the other target kind, and nothing files one yet.

    Covered anyway because `assessed` and `public_statement` are seeded and
    unwritten: the first agent that files a claim for review should find the
    queue already able to show it.
    """

    from predictelection.sql import (
        EvidenceStance,
        RecordOrigin,
        get_predicate_spec,
        new_claim,
        new_claim_assertion,
    )
    from predictelection.tests.factories import (
        make_anchor,
        make_claim_subject_and_object,
        make_research_run,
        unique,
    )

    predicate = get_predicate_spec("party_affiliation")
    subject_id, object_id = make_claim_subject_and_object(session, predicate)
    claim = new_claim(
        predicate=predicate,
        subject_id=subject_id,
        object_id=object_id,
        valid_at=datetime(2026, 1, 1, tzinfo=UTC),
        valid_at_precision=TimePrecision.DAY,
    )
    session.add(claim)
    session.flush()
    assertion = new_claim_assertion(
        session,
        claim=claim,
        evidence_anchor=make_anchor(session, excerpt="ran as a Democrat"),
        idempotency_key=unique("assertion-"),
        stance=EvidenceStance.SUPPORTS,
        origin=RecordOrigin.MODEL,
        research_run_id=make_research_run(session).id,
        asserted_by="test",
    )
    session.flush()
    task = ReviewTask(claim_assertion_id=assertion.id, reason="check the party")
    session.add(task)
    session.flush()

    view = task_view(session, task)
    labels = dict(view.target.details)
    assert view.target.kind == "claim_assertion"
    assert labels["predicate"] == "party_affiliation"
    assert labels["evidence"] == "ran as a Democrat"
    assert labels["asserted by"] == "test"
