"""The identifier namespace registry.

Entities carry identifiers from several schemes at once, and schemes retire —
eduPersonTargetedID was deprecated with a named successor and later made
obsolete, and OCD could go the same way well inside this project's life. So a
scheme is a row that can be marked deprecated, not a string literal that would
have to be found and edited across the codebase.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from predictelection.research import ScrapedEntity
from predictelection.sql import (
    EntityIdentifier,
    EntityKind,
    EntityMention,
    ExternalIdentifier,
    IdentifierNamespace,
    NAMESPACE_SPECS,
    NamespaceStatus,
    active_namespaces,
    resolve_entity_mention,
    seed_identifier_namespaces,
)
from predictelection.tests import factories as f


pytestmark = pytest.mark.postgres


def test_the_registry_is_seeded_and_idempotent(session: Session) -> None:
    assert session.scalar(select(func.count(IdentifierNamespace.namespace))) == len(
        NAMESPACE_SPECS
    )
    seed_identifier_namespaces(session)
    assert session.scalar(select(func.count(IdentifierNamespace.namespace))) == len(
        NAMESPACE_SPECS
    )


def test_ocd_outranks_wikidata_for_geography(session: Session) -> None:
    """Precedence is data, so a change of mind is an UPDATE, not a code edit."""

    ordered = [namespace.namespace for namespace in active_namespaces(session)]
    assert ordered.index("ocd-division") < ordered.index("wikidata")
    assert ordered.index("wikidata") < ordered.index("ballotpedia")


def test_an_entity_can_hold_several_identifiers(session: Session) -> None:
    """Practice is that all identifiers keep existing, none is eliminated."""

    resolved = resolve_entity_mention(
        session,
        EntityMention(
            kind=EntityKind.JURISDICTION,
            name="Michigan",
            identifiers=(
                ExternalIdentifier("wikidata", "Q1166"),
                ExternalIdentifier("ocd-division", "ocd-division/country:us/state:mi"),
            ),
        ),
    )
    session.flush()

    stored = {
        row.namespace: row.value
        for row in session.scalars(
            select(EntityIdentifier).where(
                EntityIdentifier.entity_id == resolved.entity_id
            )
        )
    }
    assert stored == {
        "wikidata": "Q1166",
        "ocd-division": "ocd-division/country:us/state:mi",
    }


def test_either_identifier_resolves_to_the_same_entity(session: Session) -> None:
    """The point of carrying several: any one of them finds the entity again."""

    original = resolve_entity_mention(
        session,
        EntityMention(
            kind=EntityKind.JURISDICTION,
            name="Michigan",
            identifiers=(
                ExternalIdentifier("wikidata", "Q1166"),
                ExternalIdentifier("ocd-division", "ocd-division/country:us/state:mi"),
            ),
        ),
    )
    by_ocd = resolve_entity_mention(
        session,
        EntityMention(
            kind=EntityKind.JURISDICTION,
            name="State of Michigan",  # a name that would not have matched
            identifiers=(
                ExternalIdentifier("ocd-division", "ocd-division/country:us/state:mi"),
            ),
        ),
    )
    assert by_ocd.entity_id == original.entity_id


def test_a_scheme_can_be_deprecated_without_touching_identifiers(
    session: Session,
) -> None:
    """The scenario this table exists for: OCD retires, nothing else moves."""

    resolved = resolve_entity_mention(
        session,
        EntityMention(
            kind=EntityKind.JURISDICTION,
            name="Michigan",
            identifiers=(
                ExternalIdentifier("ocd-division", "ocd-division/country:us/state:mi"),
            ),
        ),
    )
    session.flush()
    before = session.scalar(select(func.count(EntityIdentifier.id)))

    ocd = session.get(IdentifierNamespace, "ocd-division")
    assert ocd is not None
    ocd.status = NamespaceStatus.DEPRECATED
    ocd.superseded_by = "wikidata"
    session.flush()

    # identifiers untouched, and the entity still resolves through them
    assert session.scalar(select(func.count(EntityIdentifier.id))) == before
    again = resolve_entity_mention(
        session,
        EntityMention(
            kind=EntityKind.JURISDICTION,
            name="Michigan",
            identifiers=(
                ExternalIdentifier("ocd-division", "ocd-division/country:us/state:mi"),
            ),
        ),
    )
    assert again.entity_id == resolved.entity_id

    # but it is no longer offered for new writes
    assert "ocd-division" not in {n.namespace for n in active_namespaces(session)}


def test_deprecating_without_a_successor_is_rejected(session: Session) -> None:
    """Practice names a replacement; obsolete is the terminal state instead."""

    ocd = session.get(IdentifierNamespace, "ocd-division")
    assert ocd is not None
    ocd.status = NamespaceStatus.DEPRECATED
    with pytest.raises(IntegrityError) as error:
        session.flush()
    assert "ck_identifier_namespace_deprecated_names_a_successor" in str(error.value)


def test_scraped_entities_carry_every_identifier_offered(session: Session) -> None:
    """No namespace is hardcoded in the domain model any more."""

    mention = ScrapedEntity(
        name="Abdul El-Sayed", wikidata_id="Q28137641", fec_id="H0MI13148"
    ).as_mention(EntityKind.PERSON)

    assert {i.namespace for i in mention.identifiers} == {"wikidata", "fec"}
    resolved = resolve_entity_mention(session, mention)
    session.flush()
    assert (
        session.scalar(
            select(func.count(EntityIdentifier.id)).where(
                EntityIdentifier.entity_id == resolved.entity_id
            )
        )
        == 2
    )


def test_identifiers_can_cite_the_run_that_asserted_them(session: Session) -> None:
    """Everything else here is attributable; identifiers should be too."""

    run = f.make_research_run(session)
    entity = f.make_entity(session, kind=EntityKind.PERSON)
    session.add(
        EntityIdentifier(
            entity_id=entity.id,
            namespace="wikidata",
            value="Q28137641",
            research_run_id=run.id,
        )
    )
    session.flush()

    stored = session.scalars(
        select(EntityIdentifier).where(EntityIdentifier.entity_id == entity.id)
    ).one()
    assert stored.research_run_id == run.id
