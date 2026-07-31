"""Behavioural tests that need a real PostgreSQL.

The schema's guarantees live almost entirely in CHECK constraints, partial unique
indexes, NULLS NOT DISTINCT keys, and PostgreSQL regex and JSONB predicates.
test_sql.py compiles the DDL but never executes it, so these are the tests that
actually exercise them. Skipped when the database is unreachable unless
--require-postgres is passed; see `make test-db`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import uuid

import pytest
from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from predictelection.sql import (
    Claim,
    ClaimAssertion,
    ClaimSupersession,
    Entity,
    EntityAlias,
    EntityIdentifier,
    EntityKind,
    EntityRedirect,
    EvidenceStance,
    PollAverage,
    PollAverageEstimate,
    PollEstimate,
    PREDICATE_SPECS,
    Predicate,
    PredicateVersion,
    RecordOrigin,
    ResearchRun,
    ResearchRunStatus,
    ReviewTask,
    ReviewTaskStatus,
    SourceSnapshot,
    TimePrecision,
    VideoEvidenceLocator,
    assert_graph_integrity,
    build_evidence_anchor_fingerprint,
    check_claim_ontology,
    check_graph_integrity,
    find_entity_redirect_chains,
    get_predicate_spec,
    new_claim,
    new_claim_assertion,
    new_entity_alias,
    new_evidence_anchor,
    new_poll_estimate,
    new_poll_option,
    new_poll_revision,
    ontology_alignment_score,
    resolve_entity,
    seed_predicates,
)
from predictelection.tests import factories as f


pytestmark = pytest.mark.postgres


# --------------------------------------------------------------------------
# The schema builds and seeds at all. Nothing below can be trusted without it.
# --------------------------------------------------------------------------


def test_seeded_catalog_matches_the_python_specs(session: Session) -> None:
    assert session.scalar(select(func.count(Predicate.id))) == len(PREDICATE_SPECS)
    assert session.scalar(select(func.count(PredicateVersion.id))) == len(
        PREDICATE_SPECS
    )
    for spec in PREDICATE_SPECS:
        version = session.get(PredicateVersion, spec.predicate_version_id)
        assert version is not None
        assert version.schema_hash == spec.schema_hash
        assert version.target_kind is spec.target_kind


def test_entity_predicates_store_sql_null_not_json_null(session: Session) -> None:
    """SQLAlchemy JSONB persists None as 'null'::jsonb unless told otherwise.

    ck_predicate_version_value_contract_matches_target requires value_schema IS
    NULL for entity predicates, which JSON null does not satisfy, so getting this
    wrong makes the whole entity half of the catalog unseedable.
    """

    entity_versions = session.scalars(
        select(PredicateVersion).where(PredicateVersion.value_schema.is_(None))
    ).all()
    expected = [s for s in PREDICATE_SPECS if s.value_model is None]
    assert len(entity_versions) == len(expected)


def test_seed_predicates_is_idempotent(session: Session) -> None:
    seed_predicates(session)
    seed_predicates(session)
    assert session.scalar(select(func.count(PredicateVersion.id))) == len(
        PREDICATE_SPECS
    )


def test_seed_predicates_rejects_a_changed_contract(session: Session) -> None:
    version = session.get(
        PredicateVersion, get_predicate_spec("endorsed").predicate_version_id
    )
    assert version is not None
    session.execute(
        update(PredicateVersion)
        .where(PredicateVersion.id == version.id)
        .values(schema_hash="0" * 64)
    )
    session.expire_all()
    with pytest.raises(ValueError, match="contract changed without a version bump"):
        seed_predicates(session)


def test_seed_predicates_rejects_drifted_subject_kinds(session: Session) -> None:
    spec = get_predicate_spec("candidate_in")
    session.execute(
        insert(Entity.metadata.tables["predicate_subject_kind"]).values(
            predicate_version_id=spec.predicate_version_id,
            entity_kind=EntityKind.MARKET,
        )
    )
    with pytest.raises(ValueError, match="stored subject kinds differ"):
        seed_predicates(session)


# --------------------------------------------------------------------------
# Tier 1: the four verified bugs
# --------------------------------------------------------------------------


def test_misaligned_claim_is_kept_flagged_and_queued(session: Session) -> None:
    """Wikontic's validate-retain-flag, rather than rejecting the claim.

    A jurisdiction cannot endorse anything, but hard-excluding the triplet costs
    recall, so the claim persists and the assertion carries the violation.
    """

    spec = get_predicate_spec("endorsed")
    subject = f.make_entity(session, kind=EntityKind.JURISDICTION)
    obj = f.make_entity(session, kind=EntityKind.PERSON)
    claim = new_claim(predicate=spec, subject_id=subject.id, object_id=obj.id)
    session.add(claim)

    assertion = new_claim_assertion(
        session,
        claim=claim,
        evidence_anchor=f.make_anchor(session),
        idempotency_key=f.unique("assert-"),
        stance=EvidenceStance.SUPPORTS,
        origin=RecordOrigin.MODEL,
    )
    session.flush()

    assert assertion.ontology_aligned is False
    assert assertion.ontology_violation is not None
    assert "subject kind jurisdiction" in assertion.ontology_violation
    # the claim itself survived
    assert session.get(Claim, claim.id) is not None

    task = session.scalars(
        select(ReviewTask).where(ReviewTask.claim_assertion_id == assertion.id)
    ).one()
    assert task.status is ReviewTaskStatus.PENDING
    assert task.reason == assertion.ontology_violation


def test_aligned_claim_is_not_flagged_or_queued(session: Session) -> None:
    spec = get_predicate_spec("endorsed")
    subject_id, object_id = f.make_claim_subject_and_object(session, spec)
    claim = new_claim(predicate=spec, subject_id=subject_id, object_id=object_id)
    session.add(claim)

    assertion = new_claim_assertion(
        session,
        claim=claim,
        evidence_anchor=f.make_anchor(session),
        idempotency_key=f.unique("assert-"),
        stance=EvidenceStance.SUPPORTS,
        origin=RecordOrigin.MODEL,
    )
    session.flush()

    assert assertion.ontology_aligned is True
    assert assertion.ontology_violation is None
    assert (
        session.scalars(
            select(ReviewTask).where(ReviewTask.claim_assertion_id == assertion.id)
        ).all()
        == []
    )


def test_check_claim_ontology_reports_a_missing_entity(session: Session) -> None:
    spec = get_predicate_spec("candidate_in")
    subject = f.make_entity(session, kind=EntityKind.PERSON)
    violation = check_claim_ontology(
        session,
        predicate=spec,
        subject_id=subject.id,
        object_id=uuid.uuid4(),
    )
    assert violation is not None
    assert "does not exist" in violation


def test_ontology_alignment_score_is_scoped_to_a_run(session: Session) -> None:
    spec = get_predicate_spec("candidate_in")
    run = f.make_research_run(session)
    anchor = f.make_anchor(session)

    good_subject, good_object = f.make_claim_subject_and_object(session, spec)
    aligned_claim = new_claim(
        predicate=spec, subject_id=good_subject, object_id=good_object
    )
    session.add(aligned_claim)
    new_claim_assertion(
        session,
        claim=aligned_claim,
        evidence_anchor=anchor,
        idempotency_key=f.unique("assert-"),
        stance=EvidenceStance.SUPPORTS,
        origin=RecordOrigin.MODEL,
        research_run_id=run.id,
    )

    bad_subject = f.make_entity(session, kind=EntityKind.MARKET)
    misaligned_claim = new_claim(
        predicate=spec,
        subject_id=bad_subject.id,
        object_id=f.make_entity(session, kind=EntityKind.CONTEST).id,
    )
    session.add(misaligned_claim)
    new_claim_assertion(
        session,
        claim=misaligned_claim,
        evidence_anchor=f.make_anchor(session),
        idempotency_key=f.unique("assert-"),
        stance=EvidenceStance.SUPPORTS,
        origin=RecordOrigin.MODEL,
        research_run_id=run.id,
    )
    session.flush()

    assert ontology_alignment_score(session, research_run_id=run.id) == 0.5
    assert ontology_alignment_score(session, research_run_id=uuid.uuid4()) is None


def test_poll_estimate_cannot_straddle_two_revisions(session: Session) -> None:
    """The bug: option and sample reach poll_revision by independent FK paths."""

    first = f.make_poll_revision(session, payload={"a": 1})
    second = f.make_poll_revision(
        session, poll=first.poll, revision_number=2, payload={"a": 2}
    )

    question = f.make_poll_question(session, revision=first)
    option = new_poll_option(question=question, position=0, label="Harris")
    session.add(option)
    other_sample = f.make_poll_sample(session, revision=second)
    session.flush()

    # The Python builder refuses outright.
    with pytest.raises(ValueError, match="different revisions"):
        new_poll_estimate(
            option=option, sample=other_sample, percentage=Decimal("48.5")
        )

    # And so does the database, for anything that goes around the builder.
    session.add(
        PollEstimate(
            option_id=option.id,
            sample_id=other_sample.id,
            poll_revision_id=option.poll_revision_id,
            percentage=Decimal("48.5"),
        )
    )
    with pytest.raises(IntegrityError) as error:
        session.flush()
    assert "fk_poll_estimate_sample_same_revision" in str(error.value)


def test_poll_estimate_within_one_revision_is_accepted(session: Session) -> None:
    revision = f.make_poll_revision(session)
    question = f.make_poll_question(session, revision=revision)
    option = new_poll_option(question=question, position=0, label="Harris")
    session.add(option)
    sample = f.make_poll_sample(session, revision=revision)
    session.flush()

    estimate = new_poll_estimate(
        option=option, sample=sample, percentage=Decimal("48.5")
    )
    session.add(estimate)
    session.flush()
    assert estimate.poll_revision_id == revision.id


def test_equal_video_offsets_share_one_fingerprint(session: Session) -> None:
    """Decimal("12.5") and Decimal("12.50") are the same instant.

    Pydantic serializes Decimal to a scale-preserving string, so without
    canonicalization these hashed differently and uq_evidence_anchor_fingerprint
    stored the same evidence twice.
    """

    snapshot = f.make_snapshot(session)
    coarse = build_evidence_anchor_fingerprint(
        source_snapshot_id=snapshot.id,
        locator=VideoEvidenceLocator(start_seconds=Decimal("12.5")),
        excerpt=None,
    )
    padded = build_evidence_anchor_fingerprint(
        source_snapshot_id=snapshot.id,
        locator=VideoEvidenceLocator(start_seconds=Decimal("12.50")),
        excerpt=None,
    )
    assert coarse == padded

    session.add(
        new_evidence_anchor(
            source_snapshot_id=snapshot.id,
            locator=VideoEvidenceLocator(start_seconds=Decimal("12.5")),
        )
    )
    session.flush()
    session.add(
        new_evidence_anchor(
            source_snapshot_id=snapshot.id,
            locator=VideoEvidenceLocator(start_seconds=Decimal("12.500")),
        )
    )
    with pytest.raises(IntegrityError) as error:
        session.flush()
    assert "uq_evidence_anchor_fingerprint" in str(error.value)


def test_claim_predicate_version_relationship_is_read_only(session: Session) -> None:
    """Assigning it could only set predicate_version_id, never target_kind."""

    spec = get_predicate_spec("candidate_in")
    version = session.get(PredicateVersion, spec.predicate_version_id)
    assert version is not None
    subject_id, object_id = f.make_claim_subject_and_object(session, spec)
    claim = new_claim(predicate=spec, subject_id=subject_id, object_id=object_id)
    session.add(claim)
    session.flush()

    assert claim.predicate_version is not None
    assert claim.predicate_version.id == version.id
    relationship = Claim.__mapper__.relationships["predicate_version"]
    assert relationship.viewonly is True


def test_claim_target_kind_must_match_the_predicate_version(session: Session) -> None:
    """The composite FK, isolated from ck_claim_target_matches_payload.

    Keep a value-shaped payload (value set, object null) so the payload check
    still passes, but point it at an entity predicate. Only the composite key
    (predicate_version_id, target_kind) can catch that.
    """

    value_spec = get_predicate_spec("event_kind")
    subject = f.make_entity(session, kind=EntityKind.EVENT)
    claim = new_claim(
        predicate=value_spec,
        subject_id=subject.id,
        value={"kind": "debate"},
    )
    session.add(claim)
    session.flush()

    entity_spec = get_predicate_spec("candidate_in")
    with pytest.raises(IntegrityError) as error:
        session.execute(
            update(Claim)
            .where(Claim.id == claim.id)
            .values(predicate_version_id=entity_spec.predicate_version_id)
        )
    assert "fk_claim_predicate_version_target" in str(error.value)


# --------------------------------------------------------------------------
# Tier 2: invariants that were stated but not enforced
# --------------------------------------------------------------------------


def test_a_source_can_be_observed_twice_with_identical_bytes(
    session: Session,
) -> None:
    """ "Still said X on 2026-07-30" is evidence, so re-fetching must be storable."""

    source = f.make_source(session)
    artifact = f.make_artifact(session)
    first = f.make_snapshot(
        session,
        source=source,
        artifact=artifact,
        retrieved_at=datetime(2026, 7, 1, tzinfo=UTC),
    )
    second = f.make_snapshot(
        session,
        source=source,
        artifact=artifact,
        retrieved_at=datetime(2026, 7, 30, tzinfo=UTC),
    )
    assert first.id != second.id

    # Same instant is still one observation.
    session.add(
        SourceSnapshot(
            source_id=source.id,
            artifact_id=artifact.id,
            retrieved_at=datetime(2026, 7, 30, tzinfo=UTC),
        )
    )
    with pytest.raises(IntegrityError) as error:
        session.flush()
    assert "uq_source_snapshot_observation" in str(error.value)


def test_assertions_from_one_transaction_are_still_ordered(session: Session) -> None:
    """created_at uses now(), which is identical across a transaction."""

    spec = get_predicate_spec("candidate_in")
    subject_id, object_id = f.make_claim_subject_and_object(session, spec)
    claim = new_claim(predicate=spec, subject_id=subject_id, object_id=object_id)
    session.add(claim)
    anchor = f.make_anchor(session)

    assertions = [
        new_claim_assertion(
            session,
            claim=claim,
            evidence_anchor=anchor,
            idempotency_key=f.unique("assert-"),
            stance=EvidenceStance.SUPPORTS,
            origin=RecordOrigin.MODEL,
        )
        for _ in range(3)
    ]
    session.flush()

    timestamps = {a.created_at for a in assertions}
    sequences = [a.seq for a in assertions]
    assert len(timestamps) == 1, "now() is per-transaction, so this is the whole point"
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == 3


def test_poll_revision_payload_hash_is_derived_not_asserted(session: Session) -> None:
    revision = f.make_poll_revision(session, payload={"headline": "Harris 48"})
    # Key order must not change the hash.
    assert (
        new_poll_revision(
            payload={"b": 2, "a": 1}, poll_id=revision.poll_id
        ).payload_hash
        == new_poll_revision(
            payload={"a": 1, "b": 2}, poll_id=revision.poll_id
        ).payload_hash
    )
    with pytest.raises(ValueError, match="derived from payload"):
        new_poll_revision(payload={"a": 1}, payload_hash="0" * 64)


def test_one_sided_poll_average_bounds_are_validated(session: Session) -> None:
    """The old check parsed as A OR B OR (C AND D), so a NULL bound skipped it."""

    aggregator = f.make_entity(session, kind=EntityKind.ORGANIZATION)
    contest = f.make_entity(session, kind=EntityKind.CONTEST)
    average = PollAverage(aggregator_id=aggregator.id, contest_id=contest.id)
    session.add(average)
    session.flush()
    revision = f.make_poll_revision(session)  # for a snapshot id
    from predictelection.sql import new_poll_average_revision

    average_revision = new_poll_average_revision(
        payload={"as_of": "2026-07-30"},
        poll_average_id=average.id,
        revision_number=1,
        source_snapshot_id=revision.source_snapshot_id,
        as_of=datetime(2026, 7, 30, tzinfo=UTC),
        origin=RecordOrigin.MODEL,
    )
    session.add(average_revision)
    session.flush()

    session.add(
        PollAverageEstimate(
            poll_average_revision_id=average_revision.id,
            choice_entity_id=f.make_entity(session).id,
            percentage=Decimal("10.0"),
            lower_bound=Decimal("90.0"),
            upper_bound=None,
        )
    )
    with pytest.raises(IntegrityError) as error:
        session.flush()
    assert "ck_poll_average_estimate_bounds_order" in str(error.value)


def test_research_run_status_must_match_its_timestamps(session: Session) -> None:
    run = f.make_research_run(session)
    with pytest.raises(IntegrityError) as error:
        session.execute(
            update(ResearchRun)
            .where(ResearchRun.id == run.id)
            .values(status=ResearchRunStatus.SUCCEEDED.value, completed_at=None)
        )
    assert "ck_research_run_status_matches_outcome" in str(error.value)


def test_assertion_cannot_supersede_another_claims_assertion(
    session: Session,
) -> None:
    spec = get_predicate_spec("candidate_in")
    anchor = f.make_anchor(session)

    def make_assertion() -> ClaimAssertion:
        subject_id, object_id = f.make_claim_subject_and_object(session, spec)
        claim = new_claim(predicate=spec, subject_id=subject_id, object_id=object_id)
        session.add(claim)
        assertion = new_claim_assertion(
            session,
            claim=claim,
            evidence_anchor=anchor,
            idempotency_key=f.unique("assert-"),
            stance=EvidenceStance.SUPPORTS,
            origin=RecordOrigin.MODEL,
        )
        session.flush()
        return assertion

    first = make_assertion()
    unrelated = make_assertion()
    with pytest.raises(IntegrityError) as error:
        session.execute(
            update(ClaimAssertion)
            .where(ClaimAssertion.id == unrelated.id)
            .values(supersedes_assertion_id=first.id)
        )
    assert "fk_claim_assertion_supersedes_same_claim" in str(error.value)


def test_alias_normalization_is_derived_and_deduped(session: Session) -> None:
    entity = f.make_entity(session)
    session.add(new_entity_alias(entity_id=entity.id, name="  Abdul  El-Sayed "))
    session.flush()

    stored = session.scalars(
        select(EntityAlias).where(EntityAlias.entity_id == entity.id)
    ).one()
    assert stored.normalized_name == "abdul el-sayed"
    assert stored.name == "  Abdul  El-Sayed "

    # A different surface form of the same name collides, including on NULL
    # language, which needs NULLS NOT DISTINCT.
    session.add(EntityAlias(entity_id=entity.id, name="ABDUL EL-SAYED"))
    with pytest.raises(IntegrityError) as error:
        session.flush()
    assert "uq_entity_alias_identity" in str(error.value)


def test_identifier_namespace_is_normalized_to_satisfy_its_check(
    session: Session,
) -> None:
    entity = f.make_entity(session)
    session.add(
        EntityIdentifier(entity_id=entity.id, namespace="  WikiData ", value="Q12345")
    )
    session.flush()
    stored = session.scalars(
        select(EntityIdentifier).where(EntityIdentifier.entity_id == entity.id)
    ).one()
    assert stored.namespace == "wikidata"


def test_the_check_still_rejects_an_unnormalized_namespace(session: Session) -> None:
    """Bypassing the ORM hook must still hit ck_..._namespace_normalized."""

    entity = f.make_entity(session)
    with pytest.raises(IntegrityError) as error:
        session.execute(
            insert(EntityIdentifier).values(
                id=uuid.uuid4(),
                entity_id=entity.id,
                namespace="WikiData",
                value="Q999",
            )
        )
    assert "ck_entity_identifier_namespace_normalized" in str(error.value)


def test_resolve_entity_follows_a_redirect_chain(session: Session) -> None:
    a, b, c = (f.make_entity(session) for _ in range(3))
    session.add_all(
        [
            EntityRedirect(
                duplicate_entity_id=a.id,
                canonical_entity_id=b.id,
                reason="duplicate",
                created_by="test",
            ),
            EntityRedirect(
                duplicate_entity_id=b.id,
                canonical_entity_id=c.id,
                reason="duplicate",
                created_by="test",
            ),
        ]
    )
    session.flush()

    assert resolve_entity(session, a.id) == c.id
    assert resolve_entity(session, c.id) == c.id
    assert find_entity_redirect_chains(session) == [(a.id, b.id)]
    problems = check_graph_integrity(session)
    assert any("itself redirected" in problem for problem in problems)


def test_redirect_cycles_are_detected(session: Session) -> None:
    a, b = f.make_entity(session), f.make_entity(session)
    session.add_all(
        [
            EntityRedirect(
                duplicate_entity_id=a.id,
                canonical_entity_id=b.id,
                reason="duplicate",
                created_by="test",
            ),
            EntityRedirect(
                duplicate_entity_id=b.id,
                canonical_entity_id=a.id,
                reason="duplicate",
                created_by="test",
            ),
        ]
    )
    session.flush()

    with pytest.raises(ValueError, match="cycle"):
        resolve_entity(session, a.id)
    with pytest.raises(ValueError, match="cycle"):
        assert_graph_integrity(session)


def test_claim_supersession_cycles_are_detected(session: Session) -> None:
    spec = get_predicate_spec("candidate_in")

    def make_claim_row() -> Claim:
        subject_id, object_id = f.make_claim_subject_and_object(session, spec)
        claim = new_claim(predicate=spec, subject_id=subject_id, object_id=object_id)
        session.add(claim)
        session.flush()
        return claim

    first, second = make_claim_row(), make_claim_row()
    session.add_all(
        [
            ClaimSupersession(
                idempotency_key=f.unique("sup-"),
                predecessor_claim_id=first.id,
                successor_claim_id=second.id,
                origin=RecordOrigin.HUMAN,
                created_by="test",
                reason="correction",
            ),
            ClaimSupersession(
                idempotency_key=f.unique("sup-"),
                predecessor_claim_id=second.id,
                successor_claim_id=first.id,
                origin=RecordOrigin.HUMAN,
                created_by="test",
                reason="correction",
            ),
        ]
    )
    session.flush()

    # Each row satisfies different_claims and the unique predecessor key.
    with pytest.raises(ValueError, match="claim_supersession cycle"):
        assert_graph_integrity(session)


def test_a_clean_graph_passes_integrity(session: Session) -> None:
    f.make_entity(session)
    assert check_graph_integrity(session) == []
    assert_graph_integrity(session)


# --------------------------------------------------------------------------
# The Python-side immutability guards, through a real flush
# --------------------------------------------------------------------------


def test_immutable_rows_reject_updates(session: Session) -> None:
    entity = f.make_entity(session)
    alias = new_entity_alias(entity_id=entity.id, name="Abdul El-Sayed")
    session.add(alias)
    session.flush()

    alias.name = "changed"
    with pytest.raises(TypeError, match="immutable"):
        session.flush()


def test_immutable_rows_reject_deletes(session: Session) -> None:
    """_prevent_immutable_delete, which no test has ever reached."""

    entity = f.make_entity(session)
    alias = new_entity_alias(entity_id=entity.id, name="Abdul El-Sayed")
    session.add(alias)
    session.flush()

    session.delete(alias)
    with pytest.raises(TypeError, match="immutable"):
        session.flush()


def test_before_insert_hooks_run_on_a_real_flush(session: Session) -> None:
    """The fingerprints are derived in before_insert, not by the constructors."""

    snapshot = f.make_snapshot(session)
    anchor = new_evidence_anchor(
        source_snapshot_id=snapshot.id,
        locator={"kind": "pdf", "page_start": 3},
    )
    anchor.fingerprint = ""  # the hook must overwrite this
    session.add(anchor)
    session.flush()
    assert len(anchor.fingerprint) == 64
    assert anchor.locator_kind == "pdf"
    assert anchor.locator["page_end"] is None  # normalized by the hook

    spec = get_predicate_spec("event_kind")
    subject = f.make_entity(session, kind=EntityKind.EVENT)
    claim = Claim(
        predicate_version_id=spec.predicate_version_id,
        target_kind=spec.target_kind,
        subject_id=subject.id,
        value={"kind": "debate"},
    )
    session.add(claim)
    session.flush()
    assert len(claim.fingerprint) == 64


def test_locator_kind_must_agree_with_the_locator_json(session: Session) -> None:
    """ck_evidence_anchor_locator_kind_matches_payload uses JSONB ->>."""

    snapshot = f.make_snapshot(session)
    with pytest.raises(IntegrityError) as error:
        session.execute(
            insert(Entity.metadata.tables["evidence_anchor"]).values(
                id=uuid.uuid4(),
                source_snapshot_id=snapshot.id,
                locator_kind="web",
                locator={"kind": "full_source"},
                fingerprint="a" * 64,
            )
        )
    assert "ck_evidence_anchor_locator_kind_matches_payload" in str(error.value)


def _open_review_task(session: Session) -> ReviewTask:
    spec = get_predicate_spec("candidate_in")
    subject_id, object_id = f.make_claim_subject_and_object(session, spec)
    claim = new_claim(predicate=spec, subject_id=subject_id, object_id=object_id)
    session.add(claim)
    assertion = new_claim_assertion(
        session,
        claim=claim,
        evidence_anchor=f.make_anchor(session),
        idempotency_key=f.unique("assert-"),
        stance=EvidenceStance.SUPPORTS,
        origin=RecordOrigin.MODEL,
    )
    task = ReviewTask(claim_assertion=assertion, reason="first")
    session.add(task)
    session.flush()
    return task


def test_only_one_open_review_task_per_target(session: Session) -> None:
    task = _open_review_task(session)
    session.add(ReviewTask(claim_assertion_id=task.claim_assertion_id, reason="second"))
    with pytest.raises(IntegrityError) as error:
        session.flush()
    assert "uq_review_task_open_claim_assertion" in str(error.value)


def test_closing_a_review_task_frees_the_slot(session: Session) -> None:
    """The index is partial on status, so a completed task must not block."""

    task = _open_review_task(session)
    task.status = ReviewTaskStatus.COMPLETED
    task.completed_at = datetime.now(UTC)
    session.flush()

    session.add(ReviewTask(claim_assertion_id=task.claim_assertion_id, reason="second"))
    session.flush()
    assert session.scalar(select(func.count(ReviewTask.id))) == 2


def test_temporal_claim_shape_is_enforced_by_the_database(session: Session) -> None:
    spec = get_predicate_spec("public_statement")
    subject = f.make_entity(session, kind=EntityKind.PERSON)
    moment = datetime(2026, 6, 1, tzinfo=UTC)
    claim = new_claim(
        predicate=spec,
        subject_id=subject.id,
        value={"topic": "healthcare", "position": "supports single payer"},
        valid_from=moment,
        valid_from_precision=TimePrecision.DAY,
        valid_to=moment + timedelta(days=30),
        valid_to_precision=TimePrecision.DAY,
    )
    session.add(claim)
    session.flush()

    with pytest.raises(IntegrityError) as error:
        session.execute(
            update(Claim)
            .where(Claim.id == claim.id)
            .values(valid_at=moment, valid_at_precision=TimePrecision.DAY.value)
        )
    assert "ck_claim_point_or_interval" in str(error.value)
