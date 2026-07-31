"""Race structure, and the join it exists to make possible.

Before this, a debate pointed at a contest with no office, no stage, no party
and no candidates. The tests that matter most here are the two that cross a
source boundary: a contest described by an agent has to be the same entity the
FEC importer created, and a debate scraped afterwards has to attach to it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection.importers import FecCandidateImporter, run_import
from predictelection.research.contests import ContestKey
from predictelection.research.debates import ScrapedDebate, ingest_debate
from predictelection.research.ingestion import IngestContext
from predictelection.research.scraped import ScrapedEntity
from predictelection.research.structure import (
    ScrapedRaceStructure,
    ingest_race_structure,
)
from predictelection.sql import (
    Claim,
    ContestStage,
    Entity,
    EntityKind,
    SourceKind,
    get_predicate_spec,
)
from predictelection.tests.helpers import assert_reingestion_is_idempotent
from predictelection.tests.test_importers import FEC_TXT


pytestmark = pytest.mark.postgres


MICHIGAN = "ocd-division/country:us/state:mi"
PAGE = b"<html><body>Michigan holds a gubernatorial election in 2026.</body></html>"


def _structure(**overrides: Any) -> ScrapedRaceStructure:
    base: dict[str, Any] = {
        "source_url": "https://example.test/mi-gov-2026",
        "division_id": MICHIGAN,
        "office": "Governor",
        "cycle": 2026,
        "stage": ContestStage.PRIMARY,
        "party": "Democratic",
        "jurisdiction_name": "Michigan",
        "advances_to": ContestStage.GENERAL,
    }
    return ScrapedRaceStructure(**(base | overrides))


@pytest.fixture
def snapshot(session: Session, object_store):
    from predictelection.research.archive import SourceArchive

    return SourceArchive(session, object_store).observe(
        kind=SourceKind.WEB_PAGE,
        canonical_url="https://example.test/mi-gov-2026",
        content=PAGE,
        media_type="text/html",
        retrieved_at=datetime(2026, 3, 1, tzinfo=UTC),
    )


@pytest.fixture
def context(session: Session, snapshot) -> IngestContext:
    return IngestContext(session=session, snapshot=snapshot)


def _claims_about(session: Session, subject_id, slug: str) -> list[Claim]:
    return list(
        session.scalars(
            select(Claim).where(
                Claim.subject_id == subject_id,
                Claim.predicate_version_id
                == get_predicate_spec(slug).predicate_version_id,
            )
        )
    )


# --------------------------------------------------------------------------


def test_a_contest_stops_floating(context: IngestContext, session: Session) -> None:
    """Stage, jurisdiction, office, election and party, all joined."""

    result = ingest_race_structure(_structure(), context)
    session.flush()

    contest = result.subject_entity_id
    for slug in (
        "contest_stage",
        "contest_in_jurisdiction",
        "contest_for_office",
        "contest_of_election",
        "contest_party",
    ):
        assert len(_claims_about(session, contest, slug)) == 1, slug

    assert result.misaligned == ()
    stage = _claims_about(session, contest, "contest_stage")[0]
    assert stage.value == {"stage": "primary"}


def test_a_primary_advances_to_its_general(
    context: IngestContext, session: Session
) -> None:
    """Derived, not named.

    The successor differs from the primary only in its stage, so the ingestor
    computes it. Asking the agent for it would be asking it to name the general
    election, which is another chance to name it differently.
    """

    result = ingest_race_structure(_structure(), context)
    session.flush()

    advances = _claims_about(session, result.subject_entity_id, "advances_to")
    assert len(advances) == 1

    general = session.get(Entity, advances[0].object_id)
    assert general is not None
    assert general.kind is EntityKind.CONTEST

    # and the general it points at is genuinely the party-free general
    key = ContestKey.build(
        division=MICHIGAN, office="governor", cycle=2026, stage=ContestStage.GENERAL
    )
    from predictelection.research.structure import contest_id_for

    assert contest_id_for(context, key) == general.id


def test_the_same_race_worded_differently_is_one_contest(
    context: IngestContext, session: Session
) -> None:
    """Identity is derived, so wording cannot fork it."""

    first = ingest_race_structure(_structure(), context)
    second = ingest_race_structure(
        _structure(
            office="governor",
            party="democratic",
            jurisdiction_name="State of Michigan",
            source_url="https://other.test/mi-gov",
        ),
        context,
    )
    session.flush()

    assert first.subject_entity_id == second.subject_entity_id
    assert second.subject_created is False


def test_a_primary_and_a_general_stay_separate(
    context: IngestContext, session: Session
) -> None:
    primary = ingest_race_structure(_structure(), context)
    general = ingest_race_structure(
        _structure(stage=ContestStage.GENERAL, party=None, advances_to=None), context
    )
    session.flush()

    assert primary.subject_entity_id != general.subject_entity_id
    # but both are for the same office and the same jurisdiction
    for slug in ("contest_for_office", "contest_in_jurisdiction"):
        one = _claims_about(session, primary.subject_entity_id, slug)[0]
        other = _claims_about(session, general.subject_entity_id, slug)[0]
        assert one.object_id == other.object_id


def test_re_ingesting_structure_writes_nothing(
    context: IngestContext, session: Session
) -> None:
    """Rule 1."""

    assert_reingestion_is_idempotent(
        session, lambda: ingest_race_structure(_structure(), context)
    )


# --------------------------------------------------------------------------
# The joins across sources — what the whole thing is for


def test_the_agents_contest_is_the_importers_contest(
    context: IngestContext, session: Session, object_store
) -> None:
    """An agent and the FEC never speak, and must still mean one race.

    This is the payoff for deriving contest identity instead of naming it. The
    importer synthesised a contest from a pipe-delimited row; the agent
    described one from prose; neither knows what the other called it.
    """

    run_import(session, object_store, FecCandidateImporter(cycle=2026), raw=FEC_TXT)
    session.flush()

    described = ingest_race_structure(
        _structure(
            division_id=MICHIGAN,
            office="US Senate",
            party="Democratic",
            stage=ContestStage.PRIMARY,
        ),
        context,
    )
    session.flush()

    # the FEC's Senate candidacy points at the very contest the agent described
    candidate_in = session.scalars(
        select(Claim).where(
            Claim.predicate_version_id
            == get_predicate_spec("candidate_in").predicate_version_id,
            Claim.object_id == described.subject_entity_id,
        )
    ).all()
    assert len(candidate_in) == 1

    contests = session.scalar(
        select(func.count(Entity.id)).where(Entity.kind == EntityKind.CONTEST)
    )
    # 2 primaries from the FEC rows (senate-dem, house-dem, house-rep = 3),
    # plus the general the agent's advances_to derived
    assert contests == 4


def test_a_debate_attaches_to_a_structured_contest(
    context: IngestContext, session: Session
) -> None:
    """The roadmap's Phase 1 acceptance criterion, end to end.

    A debate ingested after structure exists must reach a contest that has a
    stage, a party, an office and at least one candidate — rather than a bare
    CONTEST entity with a name and nothing else.
    """

    structure = ingest_race_structure(_structure(), context)
    session.flush()

    key = ContestKey.build(
        division=MICHIGAN,
        office="governor",
        cycle=2026,
        stage=ContestStage.PRIMARY,
        party="democratic",
    )
    debate = ingest_debate(
        ScrapedDebate(
            source_url="https://example.test/mi-gov-debate",
            title="2026 Michigan Democratic Gubernatorial Primary Debate",
            starts_at=datetime(2026, 7, 1, tzinfo=UTC),
            participants=(ScrapedEntity(name="Abdul El-Sayed"),),
            contest=ScrapedEntity(name="Michigan Governor 2026", contest_key=str(key)),
        ),
        context,
    )
    session.flush()

    about = _claims_about(session, debate.subject_entity_id, "event_about_contest")
    assert len(about) == 1
    assert about[0].object_id == structure.subject_entity_id

    contest = structure.subject_entity_id
    assert _claims_about(session, contest, "contest_stage")
    assert _claims_about(session, contest, "contest_party")
    assert _claims_about(session, contest, "contest_for_office")


def test_structure_and_candidacies_meet_on_one_contest(
    context: IngestContext, session: Session, object_store
) -> None:
    """The join backtesting will actually use: candidates, by office.

    Nothing in this query mentions a contest by name, which is the point — it
    walks candidate_in to contest_for_office, and both sides only line up
    because the contest they share was identified rather than described.
    """

    run_import(session, object_store, FecCandidateImporter(cycle=2026), raw=FEC_TXT)
    ingest_race_structure(_structure(office="US Senate", party="Democratic"), context)
    session.flush()

    candidate_in = get_predicate_spec("candidate_in").predicate_version_id
    for_office = get_predicate_spec("contest_for_office").predicate_version_id

    candidacy = Claim.__table__.alias("candidacy")
    office_claim = Claim.__table__.alias("office_claim")
    rows = session.execute(
        select(candidacy.c.subject_id, office_claim.c.object_id)
        .join(
            office_claim,
            office_claim.c.subject_id == candidacy.c.object_id,
        )
        .where(
            candidacy.c.predicate_version_id == candidate_in,
            office_claim.c.predicate_version_id == for_office,
        )
    ).all()

    assert len(rows) == 1
    person_id, office_id = rows[0]
    person = session.get(Entity, person_id)
    office = session.get(Entity, office_id)
    assert person is not None and person.kind is EntityKind.PERSON
    assert office is not None and office.kind is EntityKind.OFFICE
