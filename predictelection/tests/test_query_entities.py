"""Search, contest description, and the typed claim payload.

Everything here was found by trying to write two pages — an entity search box
and a race timeline — against `query` and hitting a wall. The tests are the
walls, so they stay hit.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from predictelection import query
from predictelection.research.contests import ContestKey
from predictelection.sql import (
    ContestResultValue,
    ContestStage,
    EndorsementValue,
    EntityIdentifier,
    EntityKind,
    EvidenceStance,
    RecordOrigin,
    TimePrecision,
    get_predicate_spec,
    merge_entities,
    new_claim,
    new_claim_assertion,
    new_entity_alias,
)
from predictelection.tests.factories import (
    make_anchor,
    make_entity,
    make_research_run,
    unique,
)


pytestmark = pytest.mark.postgres


def _named(session: Session, name: str, kind: EntityKind = EntityKind.PERSON):
    found = make_entity(session, kind=kind, canonical_name=name)
    session.add(new_entity_alias(entity_id=found.id, name=name))
    session.flush()
    return found


def _claim(session: Session, slug: str, *, subject, object=None, value=None):
    predicate = get_predicate_spec(slug)
    claim = new_claim(
        predicate=predicate,
        subject_id=subject.id,
        object_id=object.id if object is not None else None,
        value=value,
        valid_at=datetime(2026, 1, 1, tzinfo=UTC),
        valid_at_precision=TimePrecision.DAY,
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
        asserted_by="test",
    )
    session.flush()
    return claim


# ------------------------------------------------------------------ search


def test_a_longer_name_is_not_penalised_for_being_longer(session: Session) -> None:
    """The bug that made search unusable, in one test.

    Ranking by `similarity()` puts `Crowley city` (0.67) above `David Crowley`
    (0.57), because whole-string trigram similarity charges a name for every
    extra word. `word_similarity` asks whether the query appears as words, and
    the graph-knowledge tie-break settles the rest.
    """

    candidate = _named(session, "David Crowley")
    town = _named(session, "Crowley city", EntityKind.JURISDICTION)
    contest = make_entity(session, kind=EntityKind.CONTEST)
    _claim(session, "candidate_in", subject=candidate, object=contest)

    hits = query.search_entities(session, "crowley")

    names = [hit.entity.name for hit in hits]
    assert names.index("David Crowley") < names.index("Crowley city")
    assert town.id in {hit.entity.entity_id for hit in hits}, "still offered"


def test_a_result_carries_something_to_tell_namesakes_apart(
    session: Session,
) -> None:
    """Two rows reading `Crowley city` and nothing else is not a result list.
    The real graph contains exactly that pair, in Texas and Louisiana."""

    first = _named(session, "Crowley city", EntityKind.JURISDICTION)
    second = _named(session, "Crowley city", EntityKind.JURISDICTION)
    session.add_all(
        [
            EntityIdentifier(
                entity_id=first.id,
                namespace="ocd-division",
                value="ocd-division/country:us/state:tx/place:crowley",
            ),
            EntityIdentifier(
                entity_id=second.id,
                namespace="ocd-division",
                value="ocd-division/country:us/state:la/place:crowley",
            ),
        ]
    )
    session.flush()

    contexts = {
        hit.context
        for hit in query.search_entities(session, "crowley city")
        if hit.context
    }
    assert any("state:tx" in context for context in contexts)
    assert any("state:la" in context for context in contexts)


def test_a_person_is_disambiguated_by_party_not_by_an_id(
    session: Session,
) -> None:
    """`Democratic` tells a reader which David Crowley this is. `fec:H0WI00123`
    does not, so party wins when a person has both."""

    person = _named(session, "Jane Candidate")
    party = make_entity(session, kind=EntityKind.PARTY, canonical_name="Democratic")
    session.add(EntityIdentifier(entity_id=person.id, namespace="fec", value="H0X1"))
    _claim(session, "party_affiliation", subject=person, object=party)

    (hit,) = query.search_entities(session, "jane candidate")
    assert hit.context == "Democratic"


def test_a_merged_entity_is_not_offered_as_a_result(session: Session) -> None:
    """A merge is a read-time redirect, so the alias that matched still belongs
    to the entity a reviewer merged away. Searching `berkeley` on the real graph
    returned `UC Berkeley` twice — once as itself, once as the duplicate."""

    duplicate = _named(session, "UC Berkeley", EntityKind.ORGANIZATION)
    survivor = _named(session, "UC Berkeley IGS", EntityKind.ORGANIZATION)
    merge_entities(
        session,
        duplicate_id=duplicate.id,
        canonical_id=survivor.id,
        reviewer="jack",
        reason="same institute",
    )

    hits = query.search_entities(session, "uc berkeley")

    assert [hit.entity.entity_id for hit in hits] == [survivor.id]
    assert hits[0].matched_alias == "UC Berkeley", "says why it matched"


def test_search_can_be_narrowed_to_a_kind(session: Session) -> None:
    _named(session, "Springfield", EntityKind.JURISDICTION)
    person = _named(session, "Springfield Pollster", EntityKind.ORGANIZATION)

    hits = query.search_entities(session, "springfield", kind=EntityKind.ORGANIZATION)
    assert [hit.entity.entity_id for hit in hits] == [person.id]


def test_an_empty_query_asks_nothing(session: Session) -> None:
    assert query.search_entities(session, "   ") == ()


def test_entity_follows_a_merge(session: Session) -> None:
    """A bookmarked link to an entity since merged away lands on the survivor
    rather than 404ing."""

    duplicate = _named(session, "Old Name", EntityKind.ORGANIZATION)
    survivor = _named(session, "New Name", EntityKind.ORGANIZATION)
    merge_entities(
        session,
        duplicate_id=duplicate.id,
        canonical_id=survivor.id,
        reviewer="jack",
        reason="renamed",
    )

    found = query.entity(session, duplicate.id)
    assert found is not None and found.name == "New Name"


# ----------------------------------------------------------------- contests


def _contest(session: Session, key: ContestKey):
    entity = make_entity(session, kind=EntityKind.CONTEST, canonical_name=key.label)
    session.add(
        EntityIdentifier(entity_id=entity.id, namespace="contest-key", value=str(key))
    )
    session.flush()
    return entity


WI_GOV = ContestKey.build(
    division="ocd-division/country:us/state:wi",
    office="governor",
    cycle=2026,
    stage=ContestStage.PRIMARY,
    party="Democratic",
)


def test_a_contest_describes_itself_from_its_key(session: Session) -> None:
    """Not from claims: `contest_stage`, `contest_party` and `contest_for_office`
    all have writers and **zero** rows on the real database, because the
    structure agent has not run. The key carries the same facts and cannot
    disagree with the entity it names."""

    contest = _contest(session, WI_GOV)

    detail = query.contest_detail(session, contest.id)

    assert detail is not None
    assert (detail.office, detail.cycle) == ("governor", 2026)
    assert detail.stage is ContestStage.PRIMARY
    assert detail.party == "democratic"
    assert detail.division == "ocd-division/country:us/state:wi"


def test_a_contest_links_to_its_jurisdiction_when_one_is_recorded(
    session: Session,
) -> None:
    """How a race reaches the 47k jurisdictions, and eventually a polygon."""

    contest = _contest(session, WI_GOV)
    wisconsin = make_entity(
        session, kind=EntityKind.JURISDICTION, canonical_name="Wisconsin"
    )
    session.add(
        EntityIdentifier(
            entity_id=wisconsin.id,
            namespace="ocd-division",
            value="ocd-division/country:us/state:wi",
        )
    )
    session.flush()

    detail = query.contest_detail(session, contest.id)
    assert detail is not None
    assert detail.jurisdiction is not None
    assert detail.jurisdiction.name == "Wisconsin"


def test_a_contest_minted_by_name_alone_is_not_described(session: Session) -> None:
    """Returning None rather than guessing: every field would be an invention."""

    nameless = make_entity(session, kind=EntityKind.CONTEST)
    assert query.contest_detail(session, nameless.id) is None


def test_races_are_browsable_by_division_prefix(session: Session) -> None:
    """OCD divisions nest, so a state prefix selects its statewide races and
    every district within it in one comparison. That is what the key is for."""

    _contest(session, WI_GOV)
    _contest(
        session,
        ContestKey.build(
            division="ocd-division/country:us/state:wi/cd:1",
            office="us-house",
            cycle=2026,
            stage=ContestStage.PRIMARY,
            party="Republican",
        ),
    )
    _contest(
        session,
        ContestKey.build(
            division="ocd-division/country:us/state:mi",
            office="governor",
            cycle=2026,
            stage=ContestStage.GENERAL,
        ),
    )

    wisconsin = query.contests_in(session, "ocd-division/country:us/state:wi")
    assert len(wisconsin) == 2, "the statewide race and the district one"
    assert all(
        detail.division.startswith("ocd-division/country:us/state:wi")
        for detail in wisconsin
    )

    governor = query.contests_in(
        session, "ocd-division/country:us/state:wi", office="governor"
    )
    assert [detail.office for detail in governor] == ["governor"]
    assert (
        query.contests_in(session, "ocd-division/country:us/state:wi", cycle=2024) == ()
    )


# -------------------------------------------------------------- claim value


def test_a_claim_value_arrives_as_its_declared_model(session: Session) -> None:
    """Not a dict. The predicate slug is the discriminator — the payload models
    carry no tag and cannot gain one, because the fingerprint hashes them."""

    person = _named(session, "A Winner")
    contest = make_entity(session, kind=EntityKind.CONTEST)
    _claim(
        session,
        "contest_result",
        subject=person,
        object=contest,
        value={"votes": 100, "share": "39.81", "place": 1, "won": True},
    )

    (row,) = query.claims_with(session, "contest_result")

    assert isinstance(row.value, ContestResultValue)
    assert row.value.votes == 100
    assert row.value.share == Decimal("39.81")
    assert row.value.won is True
    assert row.raw_value == {"votes": 100, "share": "39.81", "place": 1, "won": True}


def test_each_predicate_gets_its_own_payload_type(session: Session) -> None:
    """Dispatch is by slug, never by trying the union: several payloads are a
    single enum field and would validate against each other."""

    endorser = _named(session, "An Endorser")
    backed = _named(session, "The Backed")
    _claim(
        session,
        "endorsed",
        subject=endorser,
        object=backed,
        value={"strength": "full", "context": None},
    )

    (row,) = query.claims_with(session, "endorsed")
    assert isinstance(row.value, EndorsementValue)


def test_a_predicate_with_no_payload_has_no_value(session: Session) -> None:
    person = _named(session, "A Candidate")
    contest = make_entity(session, kind=EntityKind.CONTEST)
    _claim(session, "candidate_in", subject=person, object=contest)

    (row,) = query.claims_with(session, "candidate_in")
    assert row.value is None
    assert row.raw_value is None
