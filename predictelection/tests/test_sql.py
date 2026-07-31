from datetime import UTC, datetime
from typing import cast
import uuid

from pydantic import ValidationError
import pytest
from sqlalchemy import Table, create_engine, create_mock_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session, configure_mappers

from predictelection.sql import (
    Artifact,
    Base,
    Claim,
    ClaimAssertion,
    Entity,
    EntityAlias,
    EntityKind,
    EntityRedirect,
    EvidenceAnchor,
    Immutable,
    PREDICATE_SPECS,
    PollAverageRevision,
    PollRevision,
    PredicateTarget,
    PredicateVersion,
    ResearchRun,
    ReviewDecision,
    ReviewTask,
    SourceSnapshot,
    TimePrecision,
    build_claim_fingerprint,
    build_poll_payload_hash,
    get_predicate_spec,
    new_claim,
    new_evidence_anchor,
)


EXPECTED_TABLES = {
    "artifact",
    "artifact_derivation",
    "claim",
    "claim_assertion",
    "claim_supersession",
    "entity",
    "entity_alias",
    "entity_identifier",
    "entity_redirect",
    "evidence_anchor",
    "political_event_projection",
    "political_event_projection_claim",
    "poll",
    "poll_average",
    "poll_average_estimate",
    "poll_average_revision",
    "poll_estimate",
    "poll_option",
    "poll_question",
    "poll_revision",
    "poll_sample",
    "predicate",
    "predicate_object_kind",
    "predicate_subject_kind",
    "predicate_version",
    "research_run",
    "research_run_input",
    "review_decision",
    "review_task",
    "source",
    "source_snapshot",
}


def test_all_models_are_registered_and_mappable():
    configure_mappers()

    assert set(Base.metadata.tables) == EXPECTED_TABLES


def test_postgresql_schema_compiles():
    statements: list[str] = []
    dialect = postgresql.dialect()

    def record_statement(statement, *args, **kwargs):
        del args, kwargs
        statements.append(str(statement.compile(dialect=dialect)))

    engine = create_mock_engine("postgresql+psycopg://", record_statement)
    Base.metadata.create_all(engine)

    create_table_statements = [
        statement for statement in statements if "CREATE TABLE" in statement
    ]
    assert len(create_table_statements) == len(EXPECTED_TABLES)
    assert any("CREATE TABLE claim " in statement for statement in statements)
    assert any("CREATE TABLE review_decision " in statement for statement in statements)
    assert any("CREATE TABLE poll_revision " in statement for statement in statements)


def test_claims_are_source_independent_and_review_is_append_only():
    assert "source_id" not in Claim.__table__.columns
    assert "review_status" not in Claim.__table__.columns
    assert "source_snapshot_id" in EvidenceAnchor.__table__.columns
    assert "evidence_anchor_id" in ClaimAssertion.__table__.columns
    assert "research_run_id" in ClaimAssertion.__table__.columns
    assert "claim_assertion_id" in ReviewDecision.__table__.columns
    assert "claim_id" not in ReviewDecision.__table__.columns

    assertion_targets = {
        foreign_key.target_fullname
        for foreign_key in ClaimAssertion.__table__.foreign_keys
    }
    assert "claim.id" in assertion_targets
    assert "evidence_anchor.id" in assertion_targets
    assert "research_run.id" in assertion_targets

    assert issubclass(Claim, Immutable)
    assert issubclass(ClaimAssertion, Immutable)
    assert issubclass(ReviewDecision, Immutable)
    assert not issubclass(ReviewTask, Immutable)


def test_immutable_rows_reject_orm_updates():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(
        engine,
        tables=[
            cast(Table, Entity.__table__),
            cast(Table, EntityAlias.__table__),
        ],
    )

    with Session(engine) as session:
        entity = Entity(kind=EntityKind.PERSON, canonical_name="Example Person")
        alias = EntityAlias(
            entity=entity,
            name="Example",
            normalized_name="example",
            language=None,
        )
        session.add_all([entity, alias])
        session.commit()

        alias.name = "Edited in place"
        with pytest.raises(TypeError, match="immutable"):
            session.commit()


def test_source_snapshots_require_content_addressed_archives():
    assert "artifact_id" in SourceSnapshot.__table__.columns
    assert not SourceSnapshot.__table__.columns.artifact_id.nullable
    assert not Artifact.__table__.columns.sha256.nullable
    assert not Artifact.__table__.columns.storage_uri.nullable
    assert "storage_version_id" in Artifact.__table__.columns


def test_pdf_evidence_uses_validated_reproducible_locators():
    source_snapshot_id = uuid.UUID("00000000-0000-0000-0000-000000000010")

    anchor = new_evidence_anchor(
        source_snapshot_id=source_snapshot_id,
        locator={
            "kind": "pdf",
            "page_start": 3,
            "bounding_boxes": [
                {
                    "page": 3,
                    "x0": 0.1,
                    "y0": 0.2,
                    "x1": 0.8,
                    "y1": 0.3,
                }
            ],
        },
        excerpt="Likely voters were surveyed.",
    )

    assert anchor.locator_kind == "pdf"
    assert anchor.locator["page_start"] == 3
    assert len(anchor.fingerprint) == 64

    with pytest.raises(ValidationError, match="page_end"):
        new_evidence_anchor(
            source_snapshot_id=source_snapshot_id,
            locator={"kind": "pdf", "page_start": 5, "page_end": 4},
        )


def test_research_runs_do_not_require_temporal():
    assert ResearchRun.__table__.columns.workflow_id.nullable
    assert ResearchRun.__table__.columns.workflow_run_id.nullable
    assert not ResearchRun.__table__.columns.idempotency_key.nullable

    research_run_table = cast(Table, ResearchRun.__table__)
    temporal_index = next(
        index
        for index in research_run_table.indexes
        if index.name == "uq_research_run_temporal_execution"
    )
    assert temporal_index.unique
    assert temporal_index.dialect_options["postgresql"]["where"] is not None


def test_predicate_catalog_has_stable_versioned_contracts():
    keys = {(spec.slug, spec.version) for spec in PREDICATE_SPECS}
    assert len(keys) == len(PREDICATE_SPECS)

    event_kind = get_predicate_spec("event_kind")
    assert event_kind.target_kind is PredicateTarget.VALUE
    assert event_kind.value_schema is not None
    assert "kind" in event_kind.value_schema["properties"]
    assert len(event_kind.schema_hash) == 64
    assert (
        event_kind.predicate_version_id
        == get_predicate_spec("event_kind").predicate_version_id
    )

    stored_target_fk = {
        foreign_key.target_fullname for foreign_key in Claim.__table__.foreign_keys
    }
    assert "predicate_version.id" in stored_target_fk
    assert "predicate_version.target_kind" in stored_target_fk
    assert "target_kind" in PredicateVersion.__table__.columns


def test_predicate_specific_pydantic_values_are_serialized_for_jsonb():
    event_kind = get_predicate_spec("event_kind")

    assert event_kind.validate_value({"kind": "debate"}) == {"kind": "debate"}
    with pytest.raises(ValidationError):
        event_kind.validate_value({"kind": "fundraiser"})
    with pytest.raises(ValidationError):
        event_kind.validate_value({"kind": "debate", "invented": True})


def test_new_claim_validates_target_and_temporal_contract():
    subject_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    object_id = uuid.UUID("00000000-0000-0000-0000-000000000002")
    participated_in = get_predicate_spec("participated_in")

    claim = new_claim(
        predicate=participated_in,
        subject_id=subject_id,
        object_id=object_id,
    )

    assert claim.object_id == object_id
    assert claim.value is None
    assert len(claim.fingerprint) == 64

    with pytest.raises(ValueError, match="requires an object"):
        new_claim(predicate=participated_in, subject_id=subject_id)

    event_kind = get_predicate_spec("event_kind")
    with pytest.raises(ValueError, match="does not accept temporal"):
        new_claim(
            predicate=event_kind,
            subject_id=subject_id,
            value={"kind": "debate"},
            valid_at=datetime(2026, 7, 30, tzinfo=UTC),
            valid_at_precision=TimePrecision.DAY,
        )

    statement = get_predicate_spec("public_statement")
    with pytest.raises(ValueError, match="requires a temporal"):
        new_claim(
            predicate=statement,
            subject_id=subject_id,
            value={"topic": "transit", "position": "Supports expansion"},
        )


def test_claim_fingerprint_is_stable_for_equivalent_json_values():
    subject_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    predicate_version_id = get_predicate_spec("public_statement").predicate_version_id
    valid_at = datetime(2026, 7, 30, tzinfo=UTC)

    first = build_claim_fingerprint(
        predicate_version_id=predicate_version_id,
        target_kind=PredicateTarget.VALUE,
        subject_id=subject_id,
        value={"topic": "transit", "position": "supports"},
        valid_at=valid_at,
        valid_at_precision=TimePrecision.DAY,
    )
    second = build_claim_fingerprint(
        predicate_version_id=predicate_version_id,
        target_kind=PredicateTarget.VALUE,
        subject_id=subject_id,
        value={"position": "supports", "topic": "transit"},
        valid_at=valid_at,
        valid_at_precision=TimePrecision.DAY,
    )

    assert first == second
    assert len(first) == 64


def test_claim_fingerprint_validates_endpoint_precision():
    subject_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    object_id = uuid.UUID("00000000-0000-0000-0000-000000000002")
    predicate_version_id = get_predicate_spec("participated_in").predicate_version_id

    with pytest.raises(ValueError, match="valid_from_precision"):
        build_claim_fingerprint(
            predicate_version_id=predicate_version_id,
            target_kind=PredicateTarget.ENTITY,
            subject_id=subject_id,
            object_id=object_id,
            valid_from=datetime(2026, 7, 30, tzinfo=UTC),
        )

    with pytest.raises(ValueError, match="timezone-aware"):
        build_claim_fingerprint(
            predicate_version_id=predicate_version_id,
            target_kind=PredicateTarget.ENTITY,
            subject_id=subject_id,
            object_id=object_id,
            valid_at=datetime(2026, 7, 30),
            valid_at_precision=TimePrecision.DAY,
        )


def test_poll_revisions_and_entity_redirects_preserve_history():
    assert "supersedes_revision_id" in PollRevision.__table__.columns
    assert "payload_hash" in PollRevision.__table__.columns
    assert "source_snapshot_id" in PollRevision.__table__.columns
    assert "supersedes_revision_id" in PollAverageRevision.__table__.columns
    entity_redirect_table = cast(Table, EntityRedirect.__table__)
    assert "duplicate_entity_id" in entity_redirect_table.primary_key.columns

    first = build_poll_payload_hash({"questions": [{"text": "Vote?"}], "n": 500})
    second = build_poll_payload_hash({"n": 500, "questions": [{"text": "Vote?"}]})
    assert first == second
