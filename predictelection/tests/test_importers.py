"""Importers against real PostgreSQL and MinIO.

Fixture bytes rather than the live FEC and OCD files: the network is the one
dependency whose behaviour is not under test, and a 100,000-row download in a
unit test is nobody's friend. Everything else is real — the file is genuinely
archived, the claims genuinely cite it, and re-running genuinely writes nothing.

Re-ingestion is the mandatory test for every writer here. Running twice and
asserting the counts do not move is the single most common way the graph breaks.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection.importers import (
    FecCandidateImporter,
    OcdImporter,
    run_import,
)
from predictelection.importers.fec import candidacies_in
from predictelection.tests.helpers import assert_reingestion_is_idempotent
from predictelection.research.contests import CONTEST_KEY_NAMESPACE, ContestKey
from predictelection.sql import (
    Claim,
    Entity,
    EntityIdentifier,
    EntityKind,
    EntityMention,
    EvidenceAnchor,
    ExternalIdentifier,
    JsonEvidenceLocator,
    RecordOrigin,
    find_entities,
    get_predicate_spec,
    resolve_entity_mention,
)


pytestmark = pytest.mark.postgres


OCD_CSV = b"""id,name
ocd-division/country:us,United States
ocd-division/country:us/state:mi,Michigan
ocd-division/country:us/state:mi/cd:11,Michigan's 11th congressional district
ocd-division/country:us/state:mi/county:wayne,Wayne County
ocd-division/country:us/state:mi/place:detroit,Detroit
ocd-division/country:us/state:mi/county:wayne/precinct:1,Wayne County Precinct 1
ocd-division/country:us/state:mi/county:wayne/precinct:2,Wayne County Precinct 2
"""

# Real column order, pipe-delimited, no header — as the FEC publishes it.
FEC_TXT = (
    b"S6MI00179|SLOTKIN, ELISSA|DEM|2026|MI|S|00|I|C|C00693234|||LANSING|MI|48906\n"
    b"H6MI11100|STEVENS, HALEY|DEM|2026|MI|H|11|I|C|C00580456|||TROY|MI|48084\n"
    b"H6MI11200|SMITH, JOHN|REP|2026|MI|H|11|C|C|C00580999|||TROY|MI|48084\n"
    b"S6MI00999|PRIOR, PAT|REP|2024|MI|S|00|C|P|C00111111|||DETROIT|MI|48201\n"
    b"H6MI99999|WITHDRAWN, WANDA|DEM|2026|MI|H|11|C|F|C00222222|||TROY|MI|48084\n"
)


# --------------------------------------------------------------------------
# OCD


def test_ocd_gives_every_jurisdiction_an_exact_identifier(
    session: Session, object_store
) -> None:
    result = run_import(session, object_store, OcdImporter(), raw=OCD_CSV)

    assert result.rows_read == 5
    # both precincts, dropped on purpose and counted rather than vanishing
    assert result.rows_skipped == 2

    michigan = session.scalars(
        select(Entity)
        .join(EntityIdentifier, EntityIdentifier.entity_id == Entity.id)
        .where(EntityIdentifier.value == "ocd-division/country:us/state:mi")
    ).one()
    assert michigan.kind is EntityKind.JURISDICTION
    assert michigan.canonical_name == "Michigan"


def test_a_jurisdiction_resolves_by_id_under_a_different_name(
    session: Session, object_store
) -> None:
    """The whole point of importing OCD.

    A later source calling Michigan something else must land on the entity this
    import created, without anyone reconciling names afterwards.
    """

    run_import(session, object_store, OcdImporter(), raw=OCD_CSV)
    imported = [
        match
        for match in find_entities(
            session, name="Michigan", kind=EntityKind.JURISDICTION
        )
        if match.canonical_name == "Michigan"
    ]
    assert len(imported) == 1

    # a poll CSV that only knows the abbreviation, and a results file that
    # writes it out longhand — same ID, so the same entity
    for alias in ("MI", "State of Michigan"):
        resolved = resolve_entity_mention(
            session,
            EntityMention(
                kind=EntityKind.JURISDICTION,
                name=alias,
                identifiers=(
                    ExternalIdentifier(
                        namespace="ocd-division",
                        value="ocd-division/country:us/state:mi",
                    ),
                ),
            ),
        )
        assert resolved.entity_id == imported[0].entity_id
        assert resolved.created is False


def test_reimporting_ocd_writes_nothing(session: Session, object_store) -> None:
    """Rule 1: run it twice, assert the graph does not move."""

    assert_reingestion_is_idempotent(
        session, lambda: run_import(session, object_store, OcdImporter(), raw=OCD_CSV)
    )


# --------------------------------------------------------------------------
# FEC


def test_fec_records_candidacies_and_party(session: Session, object_store) -> None:
    result = run_import(
        session, object_store, FecCandidateImporter(cycle=2026), raw=FEC_TXT
    )

    # 3 usable rows: a 2024 cycle row and a withdrawn filing are excluded
    assert result.rows_read == 3
    assert result.rows_skipped == 2
    assert result.rows_failed == 0
    # candidate_in + party_affiliation for each
    assert len(result.recorded) == 6
    assert result.alignment == 1.0
    assert result.misaligned_count == 0

    candidate_in = session.scalars(
        select(func.count(Claim.id)).where(
            Claim.predicate_version_id
            == get_predicate_spec("candidate_in").predicate_version_id
        )
    ).one()
    assert candidate_in == 3


def test_fec_puts_the_house_district_in_the_division(session: Session) -> None:
    """So a House contest joins to the jurisdiction OCD created for cd:11."""

    keys = dict(candidacies_in(FEC_TXT, 2026))
    assert str(keys["H6MI11100"]) == (
        "ocd-division/country:us/state:mi/cd:11/us-house/2026/primary/democratic"
    )
    assert str(keys["S6MI00179"]) == (
        "ocd-division/country:us/state:mi/us-senate/2026/primary/democratic"
    )


def test_two_parties_candidates_are_in_different_contests(
    session: Session, object_store
) -> None:
    """Filing for an office is not winning the nomination.

    Putting both parties' filers in one contest would collapse the primary and
    the general, which have different candidates, polls and outcomes.
    """

    run_import(session, object_store, FecCandidateImporter(cycle=2026), raw=FEC_TXT)

    contests = session.scalars(
        select(Entity).where(Entity.kind == EntityKind.CONTEST)
    ).all()
    keys = {
        ContestKey.parse(identifier.value)
        for contest in contests
        for identifier in session.scalars(
            select(EntityIdentifier).where(
                EntityIdentifier.entity_id == contest.id,
                EntityIdentifier.namespace == CONTEST_KEY_NAMESPACE,
            )
        )
    }
    house_parties = {key.party for key in keys if key.office == "us-house"}
    assert house_parties == {"democratic", "republican"}


def test_the_fec_contest_is_the_one_the_key_names(
    session: Session, object_store
) -> None:
    """An agent deriving the same key must reach the same CONTEST entity.

    This is what makes the importer and the structure agent able to describe
    one race without ever agreeing on its name.
    """

    run_import(session, object_store, FecCandidateImporter(cycle=2026), raw=FEC_TXT)

    key = ContestKey.parse(
        "ocd-division/country:us/state:mi/us-senate/2026/primary/democratic"
    )
    from_agent = resolve_entity_mention(
        session,
        EntityMention(
            kind=EntityKind.CONTEST,
            # deliberately nothing like the label the importer minted
            name="2026 Democratic primary for the United States Senate in Michigan",
            identifiers=(
                ExternalIdentifier(namespace=CONTEST_KEY_NAMESPACE, value=str(key)),
            ),
        ),
    )
    assert from_agent.created is False


def test_reimporting_fec_writes_nothing(session: Session, object_store) -> None:
    """Rule 1, for the importer that actually writes claims.

    The subtle half is the evidence anchor. A snapshot is an observation keyed
    on when it was taken, so a naive retry archives the identical bytes again,
    gets a new anchor, and writes a second assertion for every row — claims
    unchanged, assertions doubled, and nothing looks wrong until the counts are
    checked.
    """

    runs = []

    def once():
        runs.append(
            run_import(
                session, object_store, FecCandidateImporter(cycle=2026), raw=FEC_TXT
            )
        )

    assert_reingestion_is_idempotent(session, once)

    first, second = runs
    # the same run, citing the same bytes, so nothing new was asserted
    assert second.research_run_id == first.research_run_id
    assert second.source_snapshot_id == first.source_snapshot_id
    assert second.claims_created == 0


def test_an_updated_file_is_new_research_not_a_retry(
    session: Session, object_store
) -> None:
    """The other half: a re-run must not be a no-op when the file has changed.

    Scoping the run to the file's content is what tells the two apart. Keying it
    on the importer and cycle alone would make every later release of the same
    file look like a retry of the first, and its new rows would never be read.
    """

    first = run_import(
        session, object_store, FecCandidateImporter(cycle=2026), raw=FEC_TXT
    )
    session.flush()

    updated = FEC_TXT + (
        b"H6MI11300|NEWCOMER, NADIA|DEM|2026|MI|H|11|C|N|C00333333|||TROY|MI|48084\n"
    )
    second = run_import(
        session, object_store, FecCandidateImporter(cycle=2026), raw=updated
    )
    session.flush()

    assert second.research_run_id != first.research_run_id
    assert second.source_snapshot_id != first.source_snapshot_id
    assert second.rows_read == 4
    # only the added candidate is new; the other three are corroborated
    assert second.claims_created == 2
    assert (
        session.scalars(
            select(func.count(Entity.id)).where(Entity.kind == EntityKind.PERSON)
        ).one()
        == 4
    )


# --------------------------------------------------------------------------
# Provenance


def test_every_imported_claim_cites_the_row_it_came_from(
    session: Session, object_store
) -> None:
    """Not just the file — the line inside it.

    One FullSourceLocator for a whole import would collapse every claim onto a
    single evidence anchor, satisfying "cites a snapshot" while losing the part
    that makes a wrong number traceable.
    """

    result = run_import(
        session, object_store, FecCandidateImporter(cycle=2026), raw=FEC_TXT
    )
    session.flush()

    anchors = session.scalars(
        select(EvidenceAnchor).where(
            EvidenceAnchor.id.in_(
                item.assertion.evidence_anchor_id for item in result.recorded
            )
        )
    ).all()

    assert len(anchors) > 1, "all claims share one anchor, so the row is not cited"
    for anchor in anchors:
        assert anchor.source_snapshot_id == result.source_snapshot_id
        assert anchor.locator["kind"] == JsonEvidenceLocator(json_pointer="").kind
        assert anchor.locator["json_pointer"].startswith("/rows/")


def test_imported_claims_are_marked_as_imports_not_model_output(
    session: Session, object_store
) -> None:
    """Review has to triage a parse bug differently from a hallucination."""

    result = run_import(
        session, object_store, FecCandidateImporter(cycle=2026), raw=FEC_TXT
    )
    assert {item.assertion.origin for item in result.recorded} == {RecordOrigin.IMPORT}


def test_one_bad_row_does_not_abort_the_file(session: Session, object_store) -> None:
    """A malformed line must cost one row, not the other ninety-nine thousand."""

    broken = FEC_TXT + b"H6MI00000|NO STATE, NELLIE|DEM|2026||H|11|C|C|C0|||X|MI|1\n"
    result = run_import(
        session, object_store, FecCandidateImporter(cycle=2026), raw=broken
    )

    assert result.rows_failed == 1
    assert result.rows_read == 4
    # the three good rows still landed
    assert len(result.recorded) == 6
