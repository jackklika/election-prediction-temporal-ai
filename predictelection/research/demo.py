"""Run one debate through the whole pipeline and print the resulting graph.

    docker compose up -d
    uv run python -m predictelection.research.demo

Uses a hand-written ScrapedDebate rather than an agent, so it exercises
archiving, entity resolution, claims, evidence, and review without needing an
API key. Re-running is safe and worth doing: nothing duplicates, which is the
property that lets a Temporal activity retry.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection.clients.sqlalchemy_engine import SqlAlchemyEngineClient
from predictelection.research.archive import SourceArchive
from predictelection.research.debates import ScrapedDebate, ScrapedEntity, ingest_debate
from predictelection.research.ingestion import IngestContext
from predictelection.sql import (
    Claim,
    ClaimAssertion,
    Entity,
    PREDICATE_SPECS,
    RecordOrigin,
    ResearchRun,
    ResearchRunStatus,
    ReviewTask,
    SourceKind,
    TimePrecision,
    create_schema,
    get_or_create,
    get_predicate_spec_by_id,
    idempotency_key,
    ontology_alignment_score,
)
from predictelection.storage import S3ObjectStore, local_minio_config


ARTICLE = b"""<html><body>
<h1>El-Sayed and Whitmer clash in Michigan governor debate</h1>
<p>The candidates met in Detroit on 15 September 2026 for 90 minutes.</p>
</body></html>"""

ARTICLE_URL = "https://example.test/mi-governor-debate-2026"

DEBATE = ScrapedDebate(
    title="2026 Michigan Gubernatorial Debate",
    source_url=ARTICLE_URL,
    starts_at=datetime(2026, 9, 15, 21, 0, tzinfo=UTC),
    starts_at_precision=TimePrecision.MINUTE,
    ends_at=datetime(2026, 9, 15, 22, 30, tzinfo=UTC),
    participants=(
        ScrapedEntity(name="Abdul El-Sayed", wikidata_id="Q28137641"),
        ScrapedEntity(name="Gretchen Whitmer", wikidata_id="Q5607626"),
    ),
    contest=ScrapedEntity(name="Michigan Governor 2026"),
    jurisdiction=ScrapedEntity(name="Michigan"),
    video_url="https://www.youtube.com/watch?v=demo",
)

RUN_KEY = idempotency_key(
    "research_run", task="find_debates", subject="michigan-governor-2026"
)


def describe(session: Session) -> None:
    """Print every claim as a sentence, with the evidence behind it."""

    names = {
        entity_id: name
        for entity_id, name in session.execute(select(Entity.id, Entity.canonical_name))
    }

    print("\nCLAIMS")
    for claim in session.scalars(select(Claim).order_by(Claim.created_at, Claim.id)):
        spec = get_predicate_spec_by_id(claim.predicate_version_id)
        subject = names.get(claim.subject_id, claim.subject_id)
        target = names[claim.object_id] if claim.object_id else claim.value
        window = ""
        if claim.valid_from is not None:
            window = f"   valid {claim.valid_from:%Y-%m-%d %H:%M}"
            window += f" .. {claim.valid_to:%H:%M}" if claim.valid_to else " ..)"
            window += f"  ({claim.valid_from_precision} precision)"
        print(f"  {subject}  --[{spec.slug}]->  {target}{window}")

        for assertion in claim.assertions:
            snapshot = assertion.evidence_anchor.source_snapshot
            flag = "" if assertion.ontology_aligned else "   << MISALIGNED"
            print(
                f"      via {assertion.origin} from "
                f"{snapshot.source.canonical_url}{flag}"
            )


def main() -> None:
    # Through the shared client rather than its own create_engine, so the demo
    # exercises the pool and echo settings the worker actually runs with.
    engine = SqlAlchemyEngineClient().engine
    create_schema(engine)

    store = S3ObjectStore(local_minio_config())
    store.ensure_bucket()

    with Session(engine) as session, session.begin():
        archive = SourceArchive(session, store)
        run, _ = get_or_create(
            session,
            ResearchRun(
                idempotency_key=RUN_KEY,
                task_type="find_debates",
                status=ResearchRunStatus.RUNNING,
            ),
            key=ResearchRun.idempotency_key == RUN_KEY,
        )

        snapshot = archive.observe(
            kind=SourceKind.WEB_PAGE,
            canonical_url=ARTICLE_URL,
            content=ARTICLE,
            media_type="text/html",
            title="El-Sayed and Whitmer clash in Michigan governor debate",
            retrieved_at=datetime(2026, 9, 16, 8, 0, tzinfo=UTC),
        )
        result = ingest_debate(
            DEBATE,
            IngestContext(
                session=session,
                snapshot=snapshot,
                archive=archive,
                research_run_id=run.id,
                asserted_by="demo",
                origin=RecordOrigin.SYSTEM,
            ),
        )

        run.status = ResearchRunStatus.SUCCEEDED
        run.completed_at = datetime.now(UTC)
        session.flush()

        score = ontology_alignment_score(session, research_run_id=run.id) or 0.0
        print(f"catalog     {len(PREDICATE_SPECS)} predicates seeded")
        print(f"archived    {snapshot.artifact.storage_uri}")
        print(f"ingested    {len(result.assertions)} assertions for this run")
        print(f"aligned     {score:.0%} of them matched the predicate's domain")
        describe(session)

    with Session(engine) as session:
        counts = {
            "entities": session.scalar(select(func.count(Entity.id))),
            "claims": session.scalar(select(func.count(Claim.id))),
            "assertions": session.scalar(select(func.count(ClaimAssertion.id))),
            "open review tasks": session.scalar(select(func.count(ReviewTask.id))),
        }
    print("\nTOTALS IN DATABASE (re-run this; they should not grow)")
    for label, total in counts.items():
        print(f"  {label:<20} {total}")


if __name__ == "__main__":
    main()
