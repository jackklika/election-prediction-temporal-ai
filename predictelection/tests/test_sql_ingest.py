"""The write path: entity resolution, deduplication, and idempotent ingestion.

These cover the operations every scraper performs, so the properties asserted
here are the ones the domain-model layer gets to rely on: resolution is total and
repeatable, re-scraping does not duplicate, and a retried activity completes.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from predictelection.sql import (
    Claim,
    ClaimAssertion,
    Entity,
    EntityAlias,
    EntityKind,
    EntityRedirect,
    EntityMention,
    EvidenceStance,
    ExternalIdentifier,
    FullSourceLocator,
    PREDICATE_SPECS,
    RecordOrigin,
    ResolutionMethod,
    ReviewTask,
    TimePrecision,
    Validity,
    get_or_create_claim,
    get_predicate_spec,
    idempotency_key,
    record_claim_from_source,
    resolve_entity_mention,
)
from predictelection.tests import factories as f


pytestmark = pytest.mark.postgres


def _person(name: str, **kwargs) -> EntityMention:
    return EntityMention(kind=EntityKind.PERSON, name=name, **kwargs)


# --------------------------------------------------------------------------
# Entity resolution
# --------------------------------------------------------------------------


def test_resolution_is_repeatable_for_the_same_name(session: Session) -> None:
    """The property a retried Temporal activity depends on."""

    first = resolve_entity_mention(session, _person("Abdul El-Sayed"))
    second = resolve_entity_mention(session, _person("Abdul El-Sayed"))

    assert first.method is ResolutionMethod.CREATED
    assert second.method is ResolutionMethod.ALIAS
    assert first.entity_id == second.entity_id
    assert session.scalar(select(func.count(Entity.id))) == 1


def test_surface_form_variants_collapse_to_one_entity(session: Session) -> None:
    canonical = resolve_entity_mention(session, _person("Abdul El-Sayed"))
    shouted = resolve_entity_mention(session, _person("  ABDUL   EL-SAYED "))

    assert shouted.entity_id == canonical.entity_id
    assert shouted.method is ResolutionMethod.ALIAS

    def keys_for(entity_id) -> set[str]:
        return set(
            session.scalars(
                select(EntityAlias.normalized_name).where(
                    EntityAlias.entity_id == entity_id
                )
            )
        )

    # uq_entity_alias_identity is keyed on the normalized name, so spellings that
    # fold together are one row rather than two
    assert keys_for(canonical.entity_id) == {"abdul el-sayed"}

    # a spelling that folds differently is a separate key, and that is how the
    # match index grows
    resolve_entity_mention(
        session, _person("Abdul El-Sayed", aliases=("Dr. Abdul El-Sayed",))
    )
    assert keys_for(canonical.entity_id) == {"abdul el-sayed", "dr. abdul el-sayed"}


def test_an_external_identifier_beats_the_name(session: Session) -> None:
    qid = (ExternalIdentifier(namespace="wikidata", value="Q123"),)
    original = resolve_entity_mention(
        session, _person("Abdul El-Sayed", identifiers=qid)
    )
    renamed = resolve_entity_mention(session, _person("A. El-Sayed", identifiers=qid))

    assert renamed.entity_id == original.entity_id
    assert renamed.method is ResolutionMethod.IDENTIFIER


def test_the_same_name_in_different_kinds_stays_separate(session: Session) -> None:
    person = resolve_entity_mention(session, _person("Washington"))
    place = resolve_entity_mention(
        session, EntityMention(kind=EntityKind.JURISDICTION, name="Washington")
    )
    assert person.entity_id != place.entity_id


def test_an_untyped_placeholder_is_promoted_not_forked(session: Session) -> None:
    """OTHER is what a scraper emits when it could not tell."""

    vague = resolve_entity_mention(
        session, EntityMention(kind=EntityKind.OTHER, name="Abdul El-Sayed")
    )
    typed = resolve_entity_mention(session, _person("Abdul El-Sayed"))

    assert typed.entity_id == vague.entity_id
    promoted = session.get(Entity, typed.entity_id)
    assert promoted is not None
    assert promoted.kind is EntityKind.PERSON


def test_an_ambiguous_name_creates_and_reports_candidates(session: Session) -> None:
    """Two real people share a name; resolution must not silently pick one."""

    first = resolve_entity_mention(session, _person("John Smith"))
    second = Entity(kind=EntityKind.PERSON, canonical_name="John Smith")
    session.add(second)
    session.flush()
    session.add(f.new_alias(second.id, "John Smith"))
    session.flush()

    third = resolve_entity_mention(session, _person("John Smith"))
    assert third.method is ResolutionMethod.CREATED
    assert third.entity_id not in {first.entity_id, second.id}
    assert set(third.ambiguous_with) == {first.entity_id, second.id}


def test_resolution_follows_a_merge(session: Session) -> None:
    duplicate = resolve_entity_mention(session, _person("Abdul El-Sayed"))
    canonical = resolve_entity_mention(session, _person("Abdul M. El-Sayed"))
    session.add(
        EntityRedirect(
            duplicate_entity_id=duplicate.entity_id,
            canonical_entity_id=canonical.entity_id,
            reason="same person",
            created_by="test",
        )
    )
    session.flush()

    again = resolve_entity_mention(session, _person("Abdul El-Sayed"))
    assert again.entity_id == canonical.entity_id


def test_an_identifier_cannot_be_stolen_by_another_entity(session: Session) -> None:
    resolve_entity_mention(
        session,
        _person("Abdul El-Sayed", identifiers=(ExternalIdentifier("wikidata", "Q1"),)),
    )
    with pytest.raises(ValueError, match="already identifies"):
        resolve_entity_mention(
            session,
            EntityMention(
                kind=EntityKind.ORGANIZATION,
                name="Some PAC",
                identifiers=(ExternalIdentifier("wikidata", "Q1"),),
            ),
        )


def test_a_blank_mention_is_rejected_at_construction(session: Session) -> None:
    with pytest.raises(ValueError, match="whitespace"):
        _person("   ")


# --------------------------------------------------------------------------
# Deduplication and idempotent ingestion
# --------------------------------------------------------------------------


def test_the_same_proposition_lands_on_one_claim(session: Session) -> None:
    spec = get_predicate_spec("candidate_in")
    subject_id, object_id = f.make_claim_subject_and_object(session, spec)

    first, created_first = get_or_create_claim(
        session, predicate=spec, subject_id=subject_id, object_id=object_id
    )
    second, created_second = get_or_create_claim(
        session, predicate=spec, subject_id=subject_id, object_id=object_id
    )

    assert created_first is True
    assert created_second is False
    assert first.id == second.id
    assert session.scalar(select(func.count(Claim.id))) == 1


def test_rescraping_corroborates_rather_than_duplicates(session: Session) -> None:
    """A second run seeing the same fact is new information, not a duplicate."""

    spec = get_predicate_spec("candidate_in")
    subject_id, object_id = f.make_claim_subject_and_object(session, spec)
    snapshot = f.make_snapshot(session)
    first_run, second_run = f.make_research_run(session), f.make_research_run(session)

    def record(run):
        return record_claim_from_source(
            session,
            predicate=spec,
            subject_id=subject_id,
            object_id=object_id,
            source_snapshot_id=snapshot.id,
            locator=FullSourceLocator(),
            research_run_id=run.id,
        )

    one, two = record(first_run).assertion, record(second_run).assertion

    assert one.claim_id == two.claim_id
    assert one.id != two.id
    assert session.scalar(select(func.count(Claim.id))) == 1
    assert session.scalar(select(func.count(ClaimAssertion.id))) == 2


def test_a_retried_activity_completes_instead_of_colliding(session: Session) -> None:
    """Same run, same evidence: the retry must be a no-op, not a unique violation."""

    spec = get_predicate_spec("candidate_in")
    subject_id, object_id = f.make_claim_subject_and_object(session, spec)
    snapshot = f.make_snapshot(session)
    run = f.make_research_run(session)

    def attempt():
        return record_claim_from_source(
            session,
            predicate=spec,
            subject_id=subject_id,
            object_id=object_id,
            source_snapshot_id=snapshot.id,
            locator=FullSourceLocator(),
            research_run_id=run.id,
        )

    first, retry = attempt().assertion, attempt().assertion

    assert first.id == retry.id
    assert session.scalar(select(func.count(ClaimAssertion.id))) == 1


def test_ingestion_flags_a_misaligned_claim_without_dropping_it(
    session: Session,
) -> None:
    spec = get_predicate_spec("endorsed")
    subject = f.make_entity(session, kind=EntityKind.JURISDICTION)
    obj = f.make_entity(session, kind=EntityKind.PERSON)

    recorded = record_claim_from_source(
        session,
        predicate=spec,
        subject_id=subject.id,
        object_id=obj.id,
        value={"strength": "full"},
        source_snapshot_id=f.make_snapshot(session).id,
        locator=FullSourceLocator(),
    )

    assert recorded.assertion.ontology_aligned is False
    assert session.scalar(select(func.count(Claim.id))) == 1
    assert session.scalar(select(func.count(ReviewTask.id))) == 1


def test_validity_builds_the_interval_columns_consistently(session: Session) -> None:
    spec = get_predicate_spec("event_occurrence")
    event = f.make_entity(session, kind=EntityKind.EVENT)
    start = datetime(2026, 9, 15, 21, 0, tzinfo=UTC)

    claim, _ = get_or_create_claim(
        session,
        predicate=spec,
        subject_id=event.id,
        value={"status": "scheduled"},
        validity=Validity.between(start, None, TimePrecision.MINUTE),
    )
    session.flush()

    assert claim.valid_from == start
    assert claim.valid_from_precision is TimePrecision.MINUTE
    assert claim.valid_to is None and claim.valid_to_precision is None
    assert claim.valid_at is None


def test_idempotency_keys_are_content_derived_and_readable(session: Session) -> None:
    same = idempotency_key("claim_assertion", claim="a", stance="supports")
    reordered = idempotency_key("claim_assertion", stance="supports", claim="a")
    different = idempotency_key("claim_assertion", claim="b", stance="supports")

    assert same == reordered, "keyword order must not change the key"
    assert same != different
    assert same.startswith("claim_assertion:")
    assert len(same) <= 255


# --------------------------------------------------------------------------
# Phase 2: the structural predicates
# --------------------------------------------------------------------------


def test_structural_predicates_are_seeded_and_usable(session: Session) -> None:
    """Without these, no query can ask which races share a geography."""

    michigan = resolve_entity_mention(
        session, EntityMention(kind=EntityKind.JURISDICTION, name="Michigan")
    )
    governor = resolve_entity_mention(
        session, EntityMention(kind=EntityKind.OFFICE, name="Governor of Michigan")
    )
    race = resolve_entity_mention(
        session, EntityMention(kind=EntityKind.CONTEST, name="Michigan Governor 2026")
    )
    election = resolve_entity_mention(
        session, EntityMention(kind=EntityKind.ELECTION, name="2026 general election")
    )
    snapshot = f.make_snapshot(session)

    links = [
        ("contest_for_office", race, governor),
        ("contest_of_election", race, election),
        ("contest_in_jurisdiction", race, michigan),
        ("office_for_jurisdiction", governor, michigan),
    ]
    for slug, subject, obj in links:
        recorded = record_claim_from_source(
            session,
            predicate=get_predicate_spec(slug),
            subject_id=subject.entity_id,
            object_id=obj.entity_id,
            source_snapshot_id=snapshot.id,
            locator=FullSourceLocator(),
        )
        assert recorded.assertion.ontology_aligned is True, slug

    # the correlation query the README wants: races decided by one geography
    contests_in_michigan = session.scalars(
        select(Claim.subject_id).where(
            Claim.predicate_version_id
            == get_predicate_spec("contest_in_jurisdiction").predicate_version_id,
            Claim.object_id == michigan.entity_id,
        )
    ).all()
    assert list(contests_in_michigan) == [race.entity_id]


def test_every_seeded_predicate_round_trips(session: Session) -> None:
    """A spec that cannot produce a valid claim is a spec nobody can use."""

    snapshot = f.make_snapshot(session)
    for spec in PREDICATE_SPECS:
        subject_id, object_id = f.make_claim_subject_and_object(session, spec)
        recorded = record_claim_from_source(
            session,
            predicate=spec,
            subject_id=subject_id,
            object_id=object_id,
            value=f.sample_value_for(spec),
            validity=f.sample_validity_for(spec),
            source_snapshot_id=snapshot.id,
            locator=FullSourceLocator(),
            excerpt=f"evidence for {spec.slug}",
            stance=EvidenceStance.SUPPORTS,
            origin=RecordOrigin.MODEL,
        )
        assert recorded.assertion.ontology_aligned is True, spec.slug


def test_an_unknown_authoritative_identifier_mints_rather_than_merging(
    session: Session,
) -> None:
    """A derived key is the definition of the entity, not a fact about it.

    Without this, two debates that share a title but not a date resolve to one
    entity by name, and the survivor ends up carrying both keys — worse than
    either a clean merge or a clean fork.
    """

    def mention(key: str) -> EntityMention:
        return EntityMention(
            kind=EntityKind.EVENT,
            name="Michigan Senate Debate",
            identifiers=(ExternalIdentifier(namespace="event-key", value=key),),
            identifiers_are_authoritative=True,
        )

    july = resolve_entity_mention(session, mention("a/debate/2026-07-07"))
    august = resolve_entity_mention(session, mention("a/debate/2026-08-07"))
    session.flush()

    assert july.entity_id != august.entity_id
    assert august.created is True


def test_a_known_authoritative_identifier_still_resolves_to_its_entity(
    session: Session,
) -> None:
    """Authoritative changes the miss path only; a hit is still a hit."""

    def mention(name: str) -> EntityMention:
        return EntityMention(
            kind=EntityKind.EVENT,
            name=name,
            identifiers=(
                ExternalIdentifier(namespace="event-key", value="a/debate/2026-07-07"),
            ),
            identifiers_are_authoritative=True,
        )

    first = resolve_entity_mention(session, mention("One wording"))
    second = resolve_entity_mention(session, mention("A quite different wording"))
    session.flush()

    assert second.entity_id == first.entity_id
    assert second.created is False


def test_an_ordinary_identifier_still_attaches_to_a_name_match(
    session: Session,
) -> None:
    """The default must stay a merge, or the OCD import forks the graph.

    "Michigan" with an ocd-division ID should adopt the Michigan a debate
    already created, not mint a second one beside it.
    """

    from_debate = resolve_entity_mention(
        session, EntityMention(kind=EntityKind.JURISDICTION, name="Michigan")
    )
    from_ocd = resolve_entity_mention(
        session,
        EntityMention(
            kind=EntityKind.JURISDICTION,
            name="Michigan",
            identifiers=(
                ExternalIdentifier(
                    namespace="ocd-division",
                    value="ocd-division/country:us/state:mi",
                ),
            ),
        ),
    )
    session.flush()

    assert from_ocd.entity_id == from_debate.entity_id
    assert from_ocd.created is False


def test_namesake_jurisdictions_with_different_identifiers_stay_apart(
    session: Session,
) -> None:
    """ "Washington township" exists in every state.

    The real OCD import merged 154 of them into one entity: the identifier
    missed (each township's ID was new), resolution fell back to the name, and
    the name matched the first township imported. A name match that contradicts
    an asserted identifier is a namesake, not the thing itself.
    """

    def township(state: str) -> EntityMention:
        return EntityMention(
            kind=EntityKind.JURISDICTION,
            name="Washington township",
            identifiers=(
                ExternalIdentifier(
                    namespace="ocd-division",
                    value=f"ocd-division/country:us/state:{state}/place:washington",
                ),
            ),
        )

    ohio = resolve_entity_mention(session, township("oh"))
    indiana = resolve_entity_mention(session, township("in"))
    session.flush()

    assert ohio.entity_id != indiana.entity_id
    assert indiana.created is True


def test_an_identifier_free_entity_is_still_adopted_by_name(
    session: Session,
) -> None:
    """The counterpart that must keep working: attaching an ID to a name.

    A debate mentions "Michigan" with no identifier; the OCD import then
    arrives with the division ID. No contradiction — the debate's Michigan has
    no competing identity — so the ID lands on the existing entity rather than
    minting a twin.
    """

    from_debate = resolve_entity_mention(
        session, EntityMention(kind=EntityKind.JURISDICTION, name="Michigan")
    )
    from_ocd = resolve_entity_mention(
        session,
        EntityMention(
            kind=EntityKind.JURISDICTION,
            name="Michigan",
            identifiers=(
                ExternalIdentifier(
                    namespace="ocd-division",
                    value="ocd-division/country:us/state:mi",
                ),
            ),
        ),
    )
    session.flush()

    assert from_ocd.entity_id == from_debate.entity_id


def test_agreement_on_an_identifier_still_merges(session: Session) -> None:
    """Same name, same identifier value: agreement, not contradiction."""

    def michigan() -> EntityMention:
        return EntityMention(
            kind=EntityKind.JURISDICTION,
            name="Michigan",
            identifiers=(
                ExternalIdentifier(
                    namespace="ocd-division",
                    value="ocd-division/country:us/state:mi",
                ),
            ),
        )

    first = resolve_entity_mention(session, michigan())
    second = resolve_entity_mention(session, michigan())
    session.flush()

    assert second.entity_id == first.entity_id
    assert second.created is False
