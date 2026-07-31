"""One negative case per database constraint.

These are the constraints test_sql.py only ever compiled. Each case builds a row
that should be rejected and asserts the specific constraint name appears in the
error, so a constraint that stops working fails its own test rather than quietly
letting bad rows through.

Cases that go through Core inserts do so on purpose: several columns are derived
by before_insert hooks or validated by new_claim, and the point here is that the
database refuses the row even when Python is bypassed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
import uuid

import pytest
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from predictelection.sql import (
    Artifact,
    ArtifactDerivation,
    ArtifactDerivationKind,
    Base,
    Claim,
    ClaimSupersession,
    EntityIdentifier,
    EntityKind,
    EntityRedirect,
    EvidenceStance,
    Poll,
    PollAverage,
    PollAverageEstimate,
    PollEstimate,
    PollOption,
    PollQuestion,
    PollSample,
    PoliticalEventProjection,
    Predicate,
    PredicateVersion,
    RecordOrigin,
    ResearchRun,
    ResearchRunStatus,
    ReviewDecision,
    ReviewOutcome,
    ReviewTask,
    ReviewTaskStatus,
    ReviewerKind,
    TimePrecision,
    get_predicate_spec,
    new_poll_option,
    new_poll_revision,
)
from predictelection.tests import factories as f


pytestmark = pytest.mark.postgres


def _claim_row(session: Session, **overrides: object) -> dict[str, object]:
    """A valid entity-target claim row, for Core inserts that override one field."""

    spec = get_predicate_spec("candidate_in")
    subject_id, object_id = f.make_claim_subject_and_object(session, spec)
    row: dict[str, object] = {
        "id": uuid.uuid4(),
        "predicate_version_id": spec.predicate_version_id,
        "target_kind": spec.target_kind.value,
        "subject_id": subject_id,
        "object_id": object_id,
        "value": None,
        "fingerprint": f"{uuid.uuid4().int:064x}"[:64],
    }
    row.update(overrides)
    return row


def _insert_claim(session: Session, **overrides: object) -> None:
    session.execute(insert(Claim).values(**_claim_row(session, **overrides)))


def _assertion_row(session: Session, **overrides: object) -> dict[str, object]:
    """A valid assertion row, with the claim it needs already inserted."""

    row = _claim_row(session)
    session.execute(insert(Claim).values(**row))
    values: dict[str, object] = {
        "id": uuid.uuid4(),
        "idempotency_key": f.unique("assert-"),
        "claim_id": row["id"],
        "evidence_anchor_id": f.make_anchor(session).id,
        "stance": EvidenceStance.SUPPORTS.value,
        "origin": RecordOrigin.MODEL.value,
    }
    values.update(overrides)
    return values


# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    constraint: str
    # returns object rather than None: builders are free to hand back whatever
    # they made, and the test ignores it
    build: Callable[[Session], object]


def _two_artifacts_sharing(session: Session, **shared: object) -> None:
    for _ in range(2):
        session.add(
            Artifact(
                sha256=shared.get("sha256") or f.unique_sha256(),
                storage_uri=shared.get("storage_uri") or f"s3://b/{f.unique()}",
                storage_version_id=None,
                byte_length=1,
            )
        )
        session.flush()


def _poll_with_external(session: Session, namespace: str | None, ext: str | None):
    poll = Poll(external_namespace=namespace, external_id=ext)
    session.add(poll)
    session.flush()
    return poll


def _review_target(session: Session):
    """An assertion id usable as a review target."""

    values = _assertion_row(session)
    session.execute(insert(Base.metadata.tables["claim_assertion"]).values(**values))
    return values["id"]


def _self_redirect(session: Session) -> None:
    entity = f.make_entity(session)
    session.add(
        EntityRedirect(
            duplicate_entity_id=entity.id,
            canonical_entity_id=entity.id,
            reason="duplicate",
            created_by="test",
        )
    )


def _redirect_without_audit(session: Session) -> None:
    duplicate, canonical = f.make_entity(session), f.make_entity(session)
    session.add(
        EntityRedirect(
            duplicate_entity_id=duplicate.id,
            canonical_entity_id=canonical.id,
            reason="",
            created_by="test",
        )
    )


def _duplicate_identifier(session: Session) -> None:
    for _ in range(2):
        session.add(
            EntityIdentifier(
                entity_id=f.make_entity(session).id,
                namespace="wikidata",
                value="Q1",
            )
        )
        session.flush()


def _assertion_superseding_itself(session: Session) -> None:
    values = _assertion_row(session)
    values["supersedes_assertion_id"] = values["id"]
    session.execute(insert(Base.metadata.tables["claim_assertion"]).values(**values))


def _two_runs_sharing_a_workflow_execution(session: Session) -> None:
    for _ in range(2):
        session.add(
            ResearchRun(
                idempotency_key=f.unique("run-"),
                task_type="t",
                workflow_id="wf-same",
                workflow_run_id="run-same",
            )
        )
        session.flush()


def _same_source_observed_twice_at_one_instant(session: Session) -> None:
    source, artifact = f.make_source(session), f.make_artifact(session)
    moment = datetime(2026, 7, 30, tzinfo=UTC)
    for _ in range(2):
        f.make_snapshot(session, source=source, artifact=artifact, retrieved_at=moment)


def _revision_reusing_a_payload(session: Session) -> None:
    first = f.make_poll_revision(session, payload={"identical": True})
    session.add(
        new_poll_revision(
            payload={"identical": True},
            poll_id=first.poll_id,
            revision_number=2,
            source_snapshot_id=f.make_snapshot(session).id,
            origin=RecordOrigin.MODEL,
        )
    )


def _revision_superseding_another_poll(session: Session) -> None:
    mine = f.make_poll_revision(session)
    someone_elses = f.make_poll_revision(session)
    session.add(
        new_poll_revision(
            payload={"other": 1},
            poll_id=mine.poll_id,
            revision_number=2,
            source_snapshot_id=f.make_snapshot(session).id,
            origin=RecordOrigin.MODEL,
            supersedes_revision_id=someone_elses.id,
        )
    )


def _estimate_without_a_measurement(session: Session) -> None:
    revision = f.make_poll_revision(session)
    option = new_poll_option(
        question=f.make_poll_question(session, revision=revision),
        position=0,
        label="Harris",
    )
    session.add(option)
    sample = f.make_poll_sample(session, revision=revision)
    session.flush()
    session.add(
        PollEstimate(
            option_id=option.id,
            sample_id=sample.id,
            poll_revision_id=revision.id,
            percentage=None,
            response_count=None,
        )
    )


def _duplicate_unnamed_series(session: Session) -> None:
    """series_name NULL twice, which only collides under NULLS NOT DISTINCT."""

    aggregator = f.make_entity(session, kind=EntityKind.ORGANIZATION)
    contest = f.make_entity(session, kind=EntityKind.CONTEST)
    for _ in range(2):
        session.add(
            PollAverage(
                aggregator_id=aggregator.id,
                contest_id=contest.id,
                series_name=None,
            )
        )
        session.flush()


def _supersession_of_itself(session: Session) -> None:
    row = _claim_row(session)
    session.execute(insert(Claim).values(**row))
    session.add(
        ClaimSupersession(
            idempotency_key=f.unique("sup-"),
            predecessor_claim_id=row["id"],
            successor_claim_id=row["id"],
            origin=RecordOrigin.HUMAN,
            created_by="t",
            reason="r",
        )
    )


def _supersession_without_audit(session: Session) -> None:
    first, second = _claim_row(session), _claim_row(session)
    session.execute(insert(Claim).values(**first))
    session.execute(insert(Claim).values(**second))
    session.add(
        ClaimSupersession(
            idempotency_key=f.unique("sup-"),
            predecessor_claim_id=first["id"],
            successor_claim_id=second["id"],
            origin=RecordOrigin.HUMAN,
            created_by="t",
            reason="",
        )
    )


def _option_borrowing_another_revision(session: Session) -> None:
    question = f.make_poll_question(session, revision=f.make_poll_revision(session))
    session.add(
        PollOption(
            question_id=question.id,
            poll_revision_id=f.make_poll_revision(session).id,
            position=0,
            label="Harris",
        )
    )


CASES: tuple[Case, ...] = (
    # ---- artifact: PostgreSQL regex, and NULLS NOT DISTINCT ----
    Case(
        "ck_artifact_sha256_lowercase_hex",
        lambda s: s.add(
            Artifact(sha256="A" * 64, storage_uri="s3://b/x", byte_length=1)
        ),
    ),
    Case(
        "ck_artifact_byte_length_nonnegative",
        lambda s: s.add(
            Artifact(sha256=f.unique_sha256(), storage_uri="s3://b/y", byte_length=-1)
        ),
    ),
    Case(
        "ck_artifact_storage_uri_nonempty",
        lambda s: s.add(
            Artifact(sha256=f.unique_sha256(), storage_uri="", byte_length=1)
        ),
    ),
    Case(
        "uq_artifact_sha256",
        lambda s: _two_artifacts_sharing(s, sha256="b" * 64),
    ),
    Case(
        "uq_artifact_storage_object",
        lambda s: _two_artifacts_sharing(s, storage_uri="s3://bucket/same"),
    ),
    # ---- entity ----
    Case(
        "ck_entity_alias_names_nonempty",
        lambda s: s.execute(
            insert(Base.metadata.tables["entity_alias"]).values(
                id=uuid.uuid4(),
                entity_id=f.make_entity(s).id,
                name="x",
                normalized_name="",
            )
        ),
    ),
    Case("uq_entity_identifier_namespace_value", _duplicate_identifier),
    Case("ck_entity_redirect_different_entities", _self_redirect),
    Case("ck_entity_redirect_audit_nonempty", _redirect_without_audit),
    # ---- predicate catalog ----
    Case(
        "ck_predicate_slug_normalized",
        lambda s: s.add(
            Predicate(id=uuid.uuid4(), slug="Bad Slug", label="x", description="x")
        ),
    ),
    Case(
        "ck_predicate_version_value_contract_matches_target",
        lambda s: s.add(
            PredicateVersion(
                id=uuid.uuid4(),
                predicate_id=get_predicate_spec("endorsed").predicate_id,
                version=99,
                target_kind="entity",
                temporal_mode="optional",
                value_model_path="a.B",
                value_schema={"type": "object"},
                schema_hash="c" * 64,
            )
        ),
    ),
    # ---- claim ----
    Case(
        "ck_claim_target_matches_payload",
        lambda s: _insert_claim(s, value={"unexpected": True}),
    ),
    Case(
        "ck_claim_valid_interval_order",
        lambda s: _insert_claim(
            s,
            valid_from=datetime(2026, 7, 2, tzinfo=UTC),
            valid_from_precision=TimePrecision.DAY.value,
            valid_to=datetime(2026, 7, 1, tzinfo=UTC),
            valid_to_precision=TimePrecision.DAY.value,
        ),
    ),
    Case(
        "ck_claim_valid_from_precision_paired",
        lambda s: _insert_claim(s, valid_from=datetime(2026, 7, 2, tzinfo=UTC)),
    ),
    Case(
        "ck_claim_fingerprint_lowercase_hex",
        lambda s: _insert_claim(s, fingerprint="Z" * 64),
    ),
    Case(
        "uq_claim_fingerprint",
        lambda s: [_insert_claim(s, fingerprint="d" * 64) for _ in range(2)],
    ),
    # ---- claim_assertion ----
    Case(
        "ck_claim_assertion_confidence_range",
        lambda s: s.execute(
            insert(Base.metadata.tables["claim_assertion"]).values(
                **_assertion_row(s, confidence=Decimal("2"))
            )
        ),
    ),
    Case(
        "ck_claim_assertion_ontology_violation_paired",
        lambda s: s.execute(
            insert(Base.metadata.tables["claim_assertion"]).values(
                **_assertion_row(s, ontology_aligned=True, ontology_violation="nope")
            )
        ),
    ),
    Case("ck_claim_assertion_does_not_supersede_self", _assertion_superseding_itself),
    # ---- claim_supersession ----
    Case("ck_claim_supersession_different_claims", _supersession_of_itself),
    Case("ck_claim_supersession_audit_nonempty", _supersession_without_audit),
    # ---- research_run ----
    Case(
        "ck_research_run_temporal_identity_complete",
        lambda s: s.add(
            ResearchRun(
                idempotency_key=f.unique("run-"),
                task_type="t",
                workflow_id="wf",
                workflow_run_id=None,
            )
        ),
    ),
    Case(
        "ck_research_run_completed_after_started",
        lambda s: s.add(
            ResearchRun(
                idempotency_key=f.unique("run-"),
                task_type="t",
                status=ResearchRunStatus.FAILED,
                error_message="boom",
                started_at=datetime(2026, 7, 30, 12, tzinfo=UTC),
                completed_at=datetime(2026, 7, 30, 11, tzinfo=UTC),
            )
        ),
    ),
    Case(
        # failed with no error_message; started_at is pinned so that
        # ck_research_run_completed_after_started cannot fire first
        "ck_research_run_status_matches_outcome",
        lambda s: s.add(
            ResearchRun(
                idempotency_key=f.unique("run-"),
                task_type="t",
                status=ResearchRunStatus.FAILED,
                started_at=datetime(2026, 7, 30, 12, tzinfo=UTC),
                completed_at=datetime(2026, 7, 30, 13, tzinfo=UTC),
                error_message=None,
            )
        ),
    ),
    Case("uq_research_run_temporal_execution", _two_runs_sharing_a_workflow_execution),
    # ---- source_snapshot ----
    Case("uq_source_snapshot_observation", _same_source_observed_twice_at_one_instant),
    # ---- artifact_derivation ----
    Case(
        "ck_artifact_derivation_different_artifacts",
        lambda s: s.add(
            ArtifactDerivation(
                parent_artifact_id=(aid := f.make_artifact(s).id),
                derived_artifact_id=aid,
                kind=ArtifactDerivationKind.OCR,
                processor_name="tesseract",
            )
        ),
    ),
    # ---- poll ----
    Case(
        "ck_poll_external_identity_complete",
        lambda s: _poll_with_external(s, "fivethirtyeight", None),
    ),
    Case(
        "uq_poll_external_identity",
        lambda s: [_poll_with_external(s, "ns", "same-id") for _ in range(2)],
    ),
    Case(
        "ck_poll_revision_revision_number_positive",
        lambda s: f.make_poll_revision(s, revision_number=0),
    ),
    Case(
        "ck_poll_revision_fieldwork_order",
        lambda s: s.add(
            new_poll_revision(
                payload={"x": 1},
                poll_id=f.make_poll_revision(s).poll_id,
                revision_number=2,
                source_snapshot_id=f.make_snapshot(s).id,
                origin=RecordOrigin.MODEL,
                fieldwork_started_on=datetime(2026, 7, 10).date(),
                fieldwork_ended_on=datetime(2026, 7, 1).date(),
            )
        ),
    ),
    Case("uq_poll_revision_payload", _revision_reusing_a_payload),
    Case("fk_poll_revision_supersedes_same_poll", _revision_superseding_another_poll),
    Case(
        "ck_poll_sample_sample_size_positive",
        lambda s: s.add(
            PollSample(
                poll_revision_id=f.make_poll_revision(s).id,
                position=0,
                label="l",
                population="p",
                sample_size=0,
            )
        ),
    ),
    Case(
        "ck_poll_sample_margin_of_error_nonnegative",
        lambda s: s.add(
            PollSample(
                poll_revision_id=f.make_poll_revision(s).id,
                position=0,
                label="l",
                population="p",
                margin_of_error=Decimal("-1"),
            )
        ),
    ),
    Case(
        "ck_poll_question_text_nonempty",
        lambda s: s.add(
            PollQuestion(
                poll_revision_id=f.make_poll_revision(s).id, position=0, text=""
            )
        ),
    ),
    Case(
        "ck_poll_option_label_nonempty",
        lambda s: s.add(
            new_poll_option(
                question=f.make_poll_question(s, revision=f.make_poll_revision(s)),
                position=0,
                label="",
            )
        ),
    ),
    Case("fk_poll_option_question_same_revision", _option_borrowing_another_revision),
    Case("ck_poll_estimate_measurement_present", _estimate_without_a_measurement),
    # ---- poll averages ----
    Case(
        "ck_poll_average_estimate_percentage_range",
        lambda s: s.add(
            PollAverageEstimate(
                poll_average_revision_id=uuid.uuid4(),
                choice_entity_id=f.make_entity(s).id,
                percentage=Decimal("101"),
            )
        ),
    ),
    Case("uq_poll_average_series", _duplicate_unnamed_series),
    # ---- review ----
    Case(
        "ck_review_decision_exactly_one_target",
        lambda s: s.add(
            ReviewDecision(
                idempotency_key=f.unique("dec-"),
                claim_assertion_id=_review_target(s),
                poll_revision_id=f.make_poll_revision(s).id,
                outcome=ReviewOutcome.ACCEPTED,
                reviewer_kind=ReviewerKind.HUMAN,
                reviewer_identifier="jack",
            )
        ),
    ),
    Case(
        "ck_review_decision_nonacceptance_has_reason",
        lambda s: s.add(
            ReviewDecision(
                idempotency_key=f.unique("dec-"),
                claim_assertion_id=_review_target(s),
                outcome=ReviewOutcome.REJECTED,
                reviewer_kind=ReviewerKind.HUMAN,
                reviewer_identifier="jack",
                reason="   ",
            )
        ),
    ),
    Case(
        "ck_review_decision_reviewer_identifier_nonempty",
        lambda s: s.add(
            ReviewDecision(
                idempotency_key=f.unique("dec-"),
                claim_assertion_id=_review_target(s),
                outcome=ReviewOutcome.ACCEPTED,
                reviewer_kind=ReviewerKind.HUMAN,
                reviewer_identifier="",
            )
        ),
    ),
    Case(
        "ck_review_task_priority_range",
        lambda s: s.add(ReviewTask(claim_assertion_id=_review_target(s), priority=101)),
    ),
    Case(
        "ck_review_task_completion_matches_status",
        lambda s: s.add(
            ReviewTask(
                claim_assertion_id=_review_target(s),
                status=ReviewTaskStatus.COMPLETED,
                completed_at=None,
            )
        ),
    ),
    Case(
        "ck_review_task_exactly_one_target",
        lambda s: s.add(ReviewTask(reason="no target")),
    ),
    # ---- event projection ----
    Case(
        "ck_political_event_projection_ends_after_starts",
        lambda s: s.add(
            PoliticalEventProjection(
                entity_id=f.make_entity(s, kind=EntityKind.EVENT).id,
                event_kind="debate",
                starts_at=datetime(2026, 7, 2, tzinfo=UTC),
                starts_at_precision=TimePrecision.DAY,
                ends_at=datetime(2026, 7, 1, tzinfo=UTC),
                ends_at_precision=TimePrecision.DAY,
            )
        ),
    ),
    Case(
        "ck_political_event_projection_starts_at_precision_paired",
        lambda s: s.add(
            PoliticalEventProjection(
                entity_id=f.make_entity(s, kind=EntityKind.EVENT).id,
                event_kind="debate",
                starts_at=datetime(2026, 7, 2, tzinfo=UTC),
                starts_at_precision=None,
            )
        ),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.constraint)
def test_constraint_rejects_its_bad_row(session: Session, case: Case) -> None:
    with pytest.raises(IntegrityError) as error:
        case.build(session)
        session.flush()
    assert case.constraint in str(error.value), (
        f"expected {case.constraint}, database raised: {error.value}"
    )


def test_every_case_names_a_real_constraint(session: Session) -> None:
    """Guards against a typo silently making a case unfalsifiable."""

    from sqlalchemy import text

    known = set(session.scalars(text("SELECT conname FROM pg_constraint")).all()) | set(
        session.scalars(text("SELECT indexname FROM pg_indexes")).all()
    )
    missing = sorted({case.constraint for case in CASES} - known)
    assert missing == []


def test_interval_claims_and_derivations_are_accepted(session: Session) -> None:
    """A positive control: the valid shapes of the rows above still insert."""

    _insert_claim(
        session,
        valid_from=datetime(2026, 7, 1, tzinfo=UTC),
        valid_from_precision=TimePrecision.DAY.value,
        valid_to=datetime(2026, 7, 2, tzinfo=UTC),
        valid_to_precision=TimePrecision.DAY.value,
    )
    parent, derived = f.make_artifact(session), f.make_artifact(session)
    session.add(
        ArtifactDerivation(
            parent_artifact_id=parent.id,
            derived_artifact_id=derived.id,
            kind=ArtifactDerivationKind.TRANSCRIPT,
            processor_name="yt-dlp",
        )
    )
    session.flush()
