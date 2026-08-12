"""Phase 3 end to end: a scraped debate becomes an attributable knowledge graph.

Runs against real PostgreSQL and real MinIO, because the point is to prove the
whole path works, not that the pieces mock cleanly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection.research import (
    IngestContext,
    ScrapedDebate,
    ScrapedEntity,
    SourceArchive,
    ingest_debate,
)
from predictelection.research.contests import EVENT_KEY_NAMESPACE
from predictelection.sql import (
    Artifact,
    ArtifactDerivationKind,
    Claim,
    ClaimAssertion,
    Entity,
    EntityIdentifier,
    EntityKind,
    ResearchRun,
    ResearchRunStatus,
    SourceKind,
    TimePrecision,
    get_predicate_spec,
    idempotency_key,
    ontology_alignment_score,
)
from predictelection.storage import ObjectNotFound
from predictelection.tests.helpers import assert_reingestion_is_idempotent


pytestmark = pytest.mark.postgres


DEBATE_PAGE = b"""<html><body>
<h1>Michigan Gubernatorial Debate</h1>
<p>Abdul El-Sayed and Gretchen Whitmer met on 15 September 2026.</p>
</body></html>"""


ARTICLE_URL = "https://example.test/michigan-debate-2026"

_DEBATE_DEFAULTS: dict[str, Any] = {
    "title": "2026 Michigan Gubernatorial Debate",
    "source_url": ARTICLE_URL,
    "starts_at": datetime(2026, 9, 15, 21, 0, tzinfo=UTC),
    "starts_at_precision": TimePrecision.MINUTE,
    "ends_at": datetime(2026, 9, 15, 22, 30, tzinfo=UTC),
    "participants": (
        ScrapedEntity(name="Abdul El-Sayed", wikidata_id="Q28137641"),
        ScrapedEntity(name="Gretchen Whitmer"),
    ),
    "contest": ScrapedEntity(name="Michigan Governor 2026"),
    "jurisdiction": ScrapedEntity(name="Michigan"),
    "video_url": "https://www.youtube.com/watch?v=abc123",
}


def _debate(**overrides: Any) -> ScrapedDebate:
    return ScrapedDebate(**(_DEBATE_DEFAULTS | overrides))


def _kind_of(session: Session, entity_id) -> EntityKind:
    entity = session.get(Entity, entity_id)
    assert entity is not None
    return entity.kind


@pytest.fixture
def archive(session: Session, object_store) -> SourceArchive:
    return SourceArchive(session, object_store)


@pytest.fixture
def snapshot(archive: SourceArchive):
    return archive.observe(
        kind=SourceKind.WEB_PAGE,
        canonical_url=ARTICLE_URL,
        content=DEBATE_PAGE,
        media_type="text/html",
        retrieved_at=datetime(2026, 9, 16, 8, 0, tzinfo=UTC),
    )


def test_archived_bytes_are_retrievable_and_content_addressed(
    archive: SourceArchive, object_store, snapshot, session: Session
) -> None:
    artifact = session.get(Artifact, snapshot.artifact_id)
    assert artifact is not None
    assert artifact.byte_length == len(DEBATE_PAGE)
    assert artifact.storage_uri.startswith("s3://")
    assert object_store.get(artifact.storage_uri) == DEBATE_PAGE

    # re-archiving identical bytes must not upload or insert again
    same = archive.artifact(content=DEBATE_PAGE, media_type="text/html")
    assert same.id == artifact.id
    assert session.scalar(select(func.count(Artifact.id))) == 1


def test_a_scraped_debate_becomes_attributable_claims(
    session: Session, snapshot
) -> None:
    run = ResearchRun(
        idempotency_key=idempotency_key("research_run", task="find_debates"),
        task_type="find_debates",
        status=ResearchRunStatus.RUNNING,
    )
    session.add(run)
    session.flush()

    result = ingest_debate(
        _debate(),
        IngestContext(session=session, snapshot=snapshot, research_run_id=run.id),
    )

    # event_kind, event_occurrence, 2 participants, contest, jurisdiction
    assert len(result.assertions) == 6
    assert result.misaligned == ()
    assert ontology_alignment_score(session, research_run_id=run.id) == 1.0

    # every claim traces back to the archived page
    for assertion in result.assertions:
        assert assertion.research_run_id == run.id
        assert assertion.evidence_anchor.source_snapshot_id == snapshot.id

    kinds = {
        _kind_of(session, entity_id)
        for entity_id in [result.subject_entity_id, *result.related_entity_ids]
    }
    assert kinds == {EntityKind.EVENT, EntityKind.PERSON}


def test_the_debate_timing_keeps_its_precision(session: Session, snapshot) -> None:
    """A source that only gave a date must not gain a false start time."""

    result = ingest_debate(
        _debate(
            starts_at=datetime(2026, 9, 15, tzinfo=UTC),
            starts_at_precision=TimePrecision.DAY,
            ends_at=None,
        ),
        IngestContext(session=session, snapshot=snapshot),
    )
    occurrence = session.scalars(
        select(Claim).where(
            Claim.subject_id == result.subject_entity_id,
            Claim.predicate_version_id
            == get_predicate_spec("event_occurrence").predicate_version_id,
        )
    ).one()

    assert occurrence.valid_from == datetime(2026, 9, 15, tzinfo=UTC)
    assert occurrence.valid_from_precision is TimePrecision.DAY
    assert occurrence.valid_to is None
    assert occurrence.value == {"status": "occurred"}


def test_rescraping_the_same_debate_is_idempotent(session: Session, snapshot) -> None:
    """The property that lets a Temporal activity retry safely."""

    results = []

    def once() -> None:
        results.append(
            ingest_debate(_debate(), IngestContext(session=session, snapshot=snapshot))
        )

    assert_reingestion_is_idempotent(session, once)

    first, second = results
    assert second.subject_entity_id == first.subject_entity_id
    assert second.related_entity_ids == first.related_entity_ids
    assert {a.id for a in second.assertions} == {a.id for a in first.assertions}


def test_a_second_source_corroborates_the_same_debate(
    session: Session, archive: SourceArchive, snapshot
) -> None:
    """Two outlets reporting one debate is one claim with two assertions."""

    ingest_debate(_debate(), IngestContext(session=session, snapshot=snapshot))
    other_snapshot = archive.observe(
        kind=SourceKind.WEB_PAGE,
        canonical_url="https://other.test/mi-debate",
        content=b"<html>A different outlet's write-up.</html>",
        media_type="text/html",
        retrieved_at=datetime(2026, 9, 17, 9, 0, tzinfo=UTC),
    )
    ingest_debate(_debate(), IngestContext(session=session, snapshot=other_snapshot))

    occurrence_claims = session.scalars(
        select(Claim.id).where(
            Claim.predicate_version_id
            == get_predicate_spec("event_occurrence").predicate_version_id
        )
    ).all()
    assert len(occurrence_claims) == 1

    assertions = session.scalars(
        select(ClaimAssertion).where(ClaimAssertion.claim_id == occurrence_claims[0])
    ).all()
    assert len(assertions) == 2
    assert len({a.evidence_anchor.source_snapshot_id for a in assertions}) == 2


def test_an_entity_id_survives_a_renamed_participant(
    session: Session, snapshot
) -> None:
    """The Wikidata QID keeps a later spelling on the same person."""

    first = ingest_debate(_debate(), IngestContext(session=session, snapshot=snapshot))
    renamed = ingest_debate(
        _debate(
            title="2026 Michigan Governor Debate (rerun coverage)",
            participants=(
                ScrapedEntity(name="Dr. Abdul El-Sayed", wikidata_id="Q28137641"),
            ),
        ),
        IngestContext(session=session, snapshot=snapshot),
    )
    assert renamed.related_entity_ids[0] == first.related_entity_ids[0]


def test_the_video_is_registered_for_later_transcription(
    session: Session, archive: SourceArchive, snapshot
) -> None:
    result = ingest_debate(
        _debate(),
        IngestContext(session=session, snapshot=snapshot, archive=archive),
    )
    assert result.registered_source_ids != ()


def test_a_transcript_keeps_its_lineage_to_the_recording(
    session: Session, archive: SourceArchive, object_store
) -> None:
    """The yt-dlp path the README wants: derived bytes, original still linked."""

    recording = archive.artifact(content=b"\x00\x01fake-video", media_type="video/mp4")
    transcript = archive.derive(
        parent=recording,
        content=b"00:00:01 Good evening and welcome to the debate.",
        kind=ArtifactDerivationKind.TRANSCRIPT,
        processor_name="yt-dlp",
        processor_version="2026.07.01",
        media_type="text/plain",
    )

    assert transcript.id != recording.id
    assert object_store.get(transcript.storage_uri).startswith(b"00:00:01")
    # recording twice does not duplicate the lineage edge
    again = archive.derive(
        parent=recording,
        content=b"00:00:01 Good evening and welcome to the debate.",
        kind=ArtifactDerivationKind.TRANSCRIPT,
        processor_name="yt-dlp",
    )
    assert again.id == transcript.id


def test_the_store_refuses_a_uri_it_does_not_own(object_store) -> None:
    """A mixed-backend database must not silently read the wrong place."""

    with pytest.raises(ValueError, match="not a s3:// URI"):
        object_store.get("gs://some-bucket/sha256/aa/bb/cc")
    with pytest.raises(ObjectNotFound):
        object_store.get(f"s3://{object_store.bucket}/sha256/aa/bb/{'a' * 64}")


MICHIGAN_OCD = "ocd-division/country:us/state:mi"


def _keyed(**overrides: Any) -> ScrapedDebate:
    """A debate whose jurisdiction carries an OCD ID, so the event can be keyed."""

    base: dict[str, Any] = {
        "jurisdiction": ScrapedEntity(name="Michigan", ocd_id=MICHIGAN_OCD),
        "starts_at": datetime(2026, 7, 7, 20, 0, tzinfo=UTC),
        "starts_at_precision": TimePrecision.MINUTE,
        "host": "WOOD TV8",
    }
    return _debate(**(base | overrides))


def test_a_rephrased_title_lands_on_the_same_event(session: Session, snapshot) -> None:
    """The measured failure: 11 event entities for 6 real debates.

    Both titles below are ones the agent actually produced for one debate
    across two runs. Identity is now the date and place, which it did not
    re-phrase.
    """

    context = IngestContext(session=session, snapshot=snapshot)
    first = ingest_debate(
        _keyed(title="Michigan Democratic U.S. Senate Primary Debate (WOOD TV8)"),
        context,
    )
    second = ingest_debate(
        _keyed(title="Michigan Democratic U.S. Senate Debate (WOOD-TV, Grand Rapids)"),
        context,
    )
    session.flush()

    assert second.subject_entity_id == first.subject_entity_id
    assert second.subject_created is False
    assert (
        session.scalar(
            select(func.count(Entity.id)).where(Entity.kind == EntityKind.EVENT)
        )
        == 1
    )


def test_two_debates_the_same_day_stay_apart_when_hosts_differ(
    session: Session, snapshot
) -> None:
    """Merging two real events loses data; only the fork is recoverable."""

    context = IngestContext(session=session, snapshot=snapshot)
    wood = ingest_debate(_keyed(title="A", host="WOOD TV8"), context)
    fox = ingest_debate(_keyed(title="B", host="Fox 2 Detroit"), context)
    session.flush()

    assert wood.subject_entity_id != fox.subject_entity_id


def test_a_debate_on_another_day_is_a_different_event(
    session: Session, snapshot
) -> None:
    context = IngestContext(session=session, snapshot=snapshot)
    july = ingest_debate(_keyed(title="A"), context)
    august = ingest_debate(
        _keyed(title="A", starts_at=datetime(2026, 8, 7, 20, 0, tzinfo=UTC)), context
    )
    session.flush()

    assert july.subject_entity_id != august.subject_entity_id


def test_without_an_ocd_jurisdiction_identity_falls_back_to_the_title(
    session: Session, snapshot
) -> None:
    """Unchanged behaviour before the OCD import has run.

    Deliberate: a key derived from an OCD ID some of the time and a name the
    rest of the time would fork on exactly the axis it exists to fix.
    """

    context = IngestContext(session=session, snapshot=snapshot)
    first = ingest_debate(
        _debate(title="One wording", jurisdiction=ScrapedEntity(name="Michigan")),
        context,
    )
    second = ingest_debate(
        _debate(title="Another wording", jurisdiction=ScrapedEntity(name="Michigan")),
        context,
    )
    session.flush()

    assert first.subject_entity_id != second.subject_entity_id


def test_a_vague_date_falls_back_to_the_title(session: Session, snapshot) -> None:
    """ "September 2026" must not merge every debate that month."""

    context = IngestContext(session=session, snapshot=snapshot)
    result = ingest_debate(
        _keyed(
            title="Vague",
            starts_at=datetime(2026, 9, 1, tzinfo=UTC),
            starts_at_precision=TimePrecision.MONTH,
        ),
        context,
    )
    session.flush()

    event = session.get(Entity, result.subject_entity_id)
    assert event is not None and event.canonical_name == "Vague"
    assert (
        session.scalars(
            select(EntityIdentifier).where(
                EntityIdentifier.entity_id == event.id,
                EntityIdentifier.namespace == EVENT_KEY_NAMESPACE,
            )
        ).first()
        is None
    )
