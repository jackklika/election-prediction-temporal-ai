"""Asking the graph questions.

Two properties carry most of the weight here, and neither is obvious from the
function signatures:

- **A predicate the reader has never heard of works anyway.** That is the whole
  claim of the ontology, and the test for it uses a predicate this package has
  no code for.
- **Review is respected.** A rejected reading of a poll must not appear on a
  timeline, and a merged pollster must appear once rather than twice.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
import uuid

import pytest
from sqlalchemy.orm import Session

from predictelection import query
from predictelection.sql import (
    AssessmentValue,
    EntityKind,
    EvidenceStance,
    Poll,
    PollEstimate,
    PollOption,
    PollQuestion,
    PollSample,
    RecordOrigin,
    ReviewOutcome,
    ReviewTask,
    TimePrecision,
    decide,
    get_predicate_spec,
    merge_entities,
    new_claim,
    new_claim_assertion,
)
from predictelection.tests.factories import (
    make_anchor,
    make_entity,
    make_poll_revision,
    make_research_run,
    unique,
)


pytestmark = pytest.mark.postgres


def _claim(
    session: Session,
    predicate_slug: str,
    *,
    subject,
    object=None,
    value: dict | None = None,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    valid_at: datetime | None = None,
    asserted_by: str = "test",
):
    predicate = get_predicate_spec(predicate_slug)
    claim = new_claim(
        predicate=predicate,
        subject_id=subject.id,
        object_id=object.id if object is not None else None,
        value=value,
        valid_at=valid_at,
        valid_at_precision=TimePrecision.DAY if valid_at else None,
        valid_from=valid_from,
        valid_from_precision=TimePrecision.DAY if valid_from else None,
        valid_to=valid_to,
        valid_to_precision=TimePrecision.DAY if valid_to else None,
    )
    session.add(claim)
    session.flush()
    new_claim_assertion(
        session,
        claim=claim,
        evidence_anchor=make_anchor(session),
        idempotency_key=unique("assertion-"),
        stance=EvidenceStance.SUPPORTS,
        origin=RecordOrigin.MODEL,
        research_run_id=make_research_run(session).id,
        asserted_by=asserted_by,
    )
    session.flush()
    return claim


def _day(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=UTC)


# ------------------------------------------------------------------ claims


def test_a_claim_row_carries_the_names_not_just_the_ids(session: Session) -> None:
    """The join everybody writes by hand, written once."""

    person = make_entity(session, kind=EntityKind.PERSON, canonical_name="A Candidate")
    contest = make_entity(session, kind=EntityKind.CONTEST, canonical_name="A Contest")
    _claim(
        session,
        "candidate_in",
        subject=person,
        object=contest,
        valid_from=_day("2026-01-01"),
    )

    (row,) = query.claims_about(session, person.id)

    assert row.predicate == "candidate_in"
    assert row.subject.name == "A Candidate"
    assert row.object is not None and row.object.name == "A Contest"
    assert row.asserted_by == ("test",)
    assert row.assertion_count == 1
    assert row.is_open


def test_both_directions_of_a_claim_are_reachable(session: Session) -> None:
    """A person's endorsements and the endorsements they received are different
    questions, and a reader that only looked at one would show half a profile."""

    endorser = make_entity(session, kind=EntityKind.PERSON, canonical_name="Endorser")
    backed = make_entity(session, kind=EntityKind.PERSON, canonical_name="Backed")
    _claim(
        session,
        "endorsed",
        subject=endorser,
        object=backed,
        value={"strength": "full"},
        valid_from=_day("2026-02-01"),
    )

    assert len(query.claims_about(session, endorser.id)) == 1
    assert len(query.claims_about(session, backed.id)) == 0
    assert len(query.claims_about(session, backed.id, as_object=True)) == 1


def test_a_predicate_this_package_has_no_code_for_still_reads(
    session: Session,
) -> None:
    """The point of the whole design: a domain nobody has written a reader for.

    `assessed` has a seeded predicate and no writer and no projection, which
    makes it the closest thing available to the donations domain that does not
    exist yet. If this passes, adding one needs no change here.
    """

    critic = make_entity(
        session, kind=EntityKind.ORGANIZATION, canonical_name="A Rater"
    )
    subject = make_entity(session, kind=EntityKind.PERSON, canonical_name="Rated")
    _claim(
        session,
        "assessed",
        subject=critic,
        object=subject,
        value={"rating": "strong", "basis": "the debate performance"},
        valid_at=_day("2026-03-01"),
    )

    (row,) = query.claims_with(session, "assessed")
    assert row.subject.name == "A Rater"
    assert isinstance(row.value, AssessmentValue)
    assert row.value.rating == "strong"


def test_asking_at_a_moment_uses_the_interval(session: Session) -> None:
    """Half-open: a claim ending on the 8th is not true on the 8th, so a
    withdrawal and a re-entry on the same day do not double-count."""

    person = make_entity(session, kind=EntityKind.PERSON)
    contest = make_entity(session, kind=EntityKind.CONTEST)
    _claim(
        session,
        "candidate_in",
        subject=person,
        object=contest,
        valid_from=_day("2026-01-01"),
        valid_to=_day("2026-07-08"),
    )

    assert query.claims_about(session, person.id, at=_day("2026-05-01"))
    assert not query.claims_about(session, person.id, at=_day("2026-07-08"))
    assert not query.claims_about(session, person.id, at=_day("2026-09-01"))


def test_a_claim_can_be_traced_to_what_said_so(session: Session) -> None:
    """The read surface's reason for existing on a project about citable facts.

    A claim you can see but not check is what the provenance model was built to
    prevent. Reached through `claim_id`, which is why every projected row keeps
    one — a result you cannot trace is a number with no source.
    """

    person = make_entity(session, kind=EntityKind.PERSON, canonical_name="Winner")
    contest = make_entity(session, kind=EntityKind.CONTEST)
    claim = _claim(
        session,
        "contest_result",
        subject=person,
        object=contest,
        value={"votes": 100, "share": None, "place": None, "won": True},
        asserted_by="import_wikipedia_results",
    )

    (result,) = query.results_for(session, contest.id)
    assert result.claim_id == claim.id

    cited = query.evidence_for(session, [result.claim_id])[result.claim_id]
    assert len(cited) == 1
    assert cited[0].asserted_by == "import_wikipedia_results"
    assert cited[0].source_url.startswith("https://")


def test_asking_for_evidence_on_nothing_costs_nothing(session: Session) -> None:
    """An empty page must not turn into `IN ()`, which Postgres rejects."""

    assert query.evidence_for(session, []) == {}


# ---------------------------------------------------------------- contests


def test_a_re_entry_is_two_stints_and_a_date_between_them_shows_neither(
    session: Session,
) -> None:
    """The Wisconsin shape, which is why `candidate_in` is not a status column."""

    person = make_entity(session, kind=EntityKind.PERSON, canonical_name="Returner")
    other = make_entity(session, kind=EntityKind.PERSON, canonical_name="Steady")
    contest = make_entity(session, kind=EntityKind.CONTEST)
    _claim(
        session,
        "candidate_in",
        subject=person,
        object=contest,
        valid_from=_day("2026-01-01"),
        valid_to=_day("2026-07-08"),
    )
    _claim(
        session,
        "candidate_in",
        subject=person,
        object=contest,
        valid_from=_day("2026-07-18"),
    )
    _claim(
        session,
        "candidate_in",
        subject=other,
        object=contest,
        valid_from=_day("2026-01-01"),
    )

    assert len(query.candidates_in(session, contest.id)) == 3  # stints, not people

    running = lambda day: sorted(  # noqa: E731 - reads better inline here
        stint.person.name
        for stint in query.candidates_in(session, contest.id, at=_day(day))
    )
    assert running("2026-05-01") == ["Returner", "Steady"]
    assert running("2026-07-10") == ["Steady"]
    assert running("2026-08-01") == ["Returner", "Steady"]


def test_results_come_back_best_first_with_the_payload_unpacked(
    session: Session,
) -> None:
    contest = make_entity(session, kind=EntityKind.CONTEST)
    for name, votes, share, won in [
        ("Runner Up", 311495, "39.33", False),
        ("Winner", 315278, "39.81", True),
    ]:
        _claim(
            session,
            "contest_result",
            subject=make_entity(session, kind=EntityKind.PERSON, canonical_name=name),
            object=contest,
            value={"votes": votes, "share": share, "place": None, "won": won},
        )

    first, second = query.results_for(session, contest.id)

    assert first.candidate.name == "Winner"
    assert first.votes == 315278
    assert first.share == Decimal("39.81")  # Decimal, not float
    assert first.won is True
    assert second.candidate.name == "Runner Up"


def test_a_result_nobody_called_is_not_reported_as_a_win(session: Session) -> None:
    """`won` is nullable because a vote table states counts, not outcomes.
    Treating NULL as a win — or as a loss — asserts something no source did."""

    contest = make_entity(session, kind=EntityKind.CONTEST)
    office = make_entity(session, kind=EntityKind.OFFICE, canonical_name="An Office")
    _claim(session, "contest_for_office", subject=contest, object=office)
    _claim(
        session,
        "contest_result",
        subject=make_entity(session, kind=EntityKind.PERSON, canonical_name="Unstated"),
        object=contest,
        value={"votes": 100, "share": None, "place": None, "won": None},
    )

    assert query.winners_by_office(session) == ()

    _claim(
        session,
        "contest_result",
        subject=make_entity(session, kind=EntityKind.PERSON, canonical_name="Called"),
        object=contest,
        value={"votes": 200, "share": None, "place": None, "won": True},
    )
    ((found_office, result),) = query.winners_by_office(session)
    assert found_office.name == "An Office"
    assert result.candidate.name == "Called"


# ------------------------------------------------------------------- polls


def _poll_point(
    session: Session,
    contest,
    pollster,
    *,
    ended_on: str,
    label: str = "overall",
    percentage: str = "40",
    poll=None,
    revision_number: int = 1,
    payload: dict | None = None,
):
    revision = make_poll_revision(
        session,
        poll=poll,
        revision_number=revision_number,
        payload=payload or {"reading": unique()},
        pollster_id=pollster.id,
        created_by="test",
        fieldwork_ended_on=datetime.fromisoformat(ended_on).date(),
    )
    question = PollQuestion(
        poll_revision_id=revision.id, contest_id=contest.id, position=0, text="q"
    )
    sample = PollSample(
        poll_revision_id=revision.id,
        position=0,
        label=label,
        population="lv",
        sample_size=800,
    )
    session.add_all([question, sample])
    session.flush()
    option = PollOption(
        question_id=question.id,
        poll_revision_id=revision.id,
        position=0,
        label="A Candidate",
    )
    session.add(option)
    session.flush()
    session.add(
        PollEstimate(
            option_id=option.id,
            sample_id=sample.id,
            poll_revision_id=revision.id,
            percentage=Decimal(percentage),
        )
    )
    session.flush()
    return revision


def test_a_timeline_is_ordered_by_fieldwork_end(session: Session) -> None:
    contest = make_entity(session, kind=EntityKind.CONTEST)
    pollster = make_entity(
        session, kind=EntityKind.ORGANIZATION, canonical_name="A Poll Co"
    )
    _poll_point(session, contest, pollster, ended_on="2026-03-01")
    _poll_point(session, contest, pollster, ended_on="2026-01-01")

    points = query.poll_timeline(session, contest.id)

    assert [str(point.fieldwork_ended_on) for point in points] == [
        "2026-01-01",
        "2026-03-01",
    ]
    assert points[0].readings[0].label == "A Candidate"
    assert points[0].reviewed is False


def test_a_rejected_reading_is_not_on_the_timeline(session: Session) -> None:
    """The rule this projection exists to enforce. "Latest revision" would show
    the wrong one; the reviewer's verdict decides, and it is not a column."""

    contest = make_entity(session, kind=EntityKind.CONTEST)
    pollster = make_entity(session, kind=EntityKind.ORGANIZATION)
    poll = Poll()
    session.add(poll)
    session.flush()

    good = _poll_point(
        session,
        contest,
        pollster,
        ended_on="2026-02-01",
        poll=poll,
        payload={"reading": "one"},
        percentage="40",
    )
    bad = _poll_point(
        session,
        contest,
        pollster,
        ended_on="2026-02-01",
        poll=poll,
        revision_number=2,
        payload={"reading": "two"},
        percentage="99",
    )
    assert len(query.poll_timeline(session, contest.id)) == 2

    task = ReviewTask(poll_revision=bad, reason="two sources disagree")
    session.add(task)
    session.flush()
    decide(
        session,
        task,
        outcome=ReviewOutcome.REJECTED,
        reviewer="jack",
        reason="does not sum to 100",
        action_token=str(uuid.uuid4()),
    )

    (point,) = query.poll_timeline(session, contest.id)
    assert point.revision_id == good.id
    assert point.reviewed is False  # the *accepted* one was never ruled on
    assert len(query.poll_timeline(session, contest.id, include_rejected=True)) == 2


def test_a_merged_pollster_is_one_series_not_two(session: Session) -> None:
    """A revision written before a merge still stores the duplicate's id — the
    redirect is a read-time indirection, not a backfill. Without resolving it
    the same firm draws two lines on one chart."""

    contest = make_entity(session, kind=EntityKind.CONTEST)
    duplicate = make_entity(
        session, kind=EntityKind.ORGANIZATION, canonical_name="Poll Co"
    )
    survivor = make_entity(
        session, kind=EntityKind.ORGANIZATION, canonical_name="Poll Co Inc"
    )
    _poll_point(session, contest, duplicate, ended_on="2026-01-01")
    _poll_point(session, contest, survivor, ended_on="2026-02-01")

    merge_entities(
        session,
        duplicate_id=duplicate.id,
        canonical_id=survivor.id,
        reviewer="jack",
        reason="same firm",
    )

    names = {
        point.pollster.name
        for point in query.poll_timeline(session, contest.id)
        if point.pollster
    }
    assert names == {"Poll Co Inc"}


# ------------------------------------------------------------------ review


def test_the_backlog_counts_the_latest_verdict_not_every_row(
    session: Session,
) -> None:
    """A reviewer who reverses themselves has made one decision, not two.

    Counting rows would report the mistake and its correction as separate
    findings — which is exactly what happened to the ARW Strategies task.
    """

    revision = make_poll_revision(session)
    task = ReviewTask(poll_revision=revision, reason="worth a look")
    session.add(task)
    session.flush()

    assert query.unreviewed(session).pending == 1
    assert query.unreviewed(session).is_clear is False

    decide(
        session,
        task,
        outcome=ReviewOutcome.REJECTED,
        reviewer="jack",
        reason="wrong",
        action_token=str(uuid.uuid4()),
    )
    query.unreviewed(session)
    # the reversal, recorded the way review_queue does it
    from predictelection.sql import decide_reading

    decide_reading(
        session,
        revision.id,
        outcome=ReviewOutcome.ACCEPTED,
        reviewer="jack",
        reason="on reflection the data is fine",
        action_token=str(uuid.uuid4()),
    )

    backlog = query.unreviewed(session)
    assert backlog.pending == 0
    assert backlog.completed == 1
    assert backlog.decisions == 1, "one target, one current verdict"
    assert (backlog.accepted, backlog.rejected) == (1, 0)
    assert backlog.is_clear
