"""Contest identity: the thing three independent sources have to agree on.

If a contest is identified by its name, an OCD importer, an FEC importer and an
agent produce three entities for one race and nothing correlates. These tests
pin the properties that stop that: the same race described differently yields
the same key, and a primary and a general never collide.

No Postgres needed — this is pure derivation.
"""

from __future__ import annotations

import pytest

from predictelection.research.contests import (
    CONTEST_KEY_NAMESPACE,
    ContestKey,
    normalize_slug,
)
from predictelection.research.scraped import ScrapedEntity
from predictelection.sql import ContestStage, EntityKind, NAMESPACE_SPECS


MICHIGAN = "ocd-division/country:us/state:mi"


def test_the_same_race_described_differently_gives_one_key() -> None:
    """The whole point. Two sources, no coordination, one identity."""

    from_importer = ContestKey.build(
        division=MICHIGAN,
        office="governor",
        cycle=2026,
        stage=ContestStage.PRIMARY,
        party="democratic",
    )
    from_agent = ContestKey.build(
        division="OCD-Division/country:us/state:mi",
        office="Governor",
        cycle=2026,
        stage=ContestStage.PRIMARY,
        party="Democratic",
    )

    assert str(from_importer) == str(from_agent)
    assert str(from_importer) == (
        "ocd-division/country:us/state:mi/governor/2026/primary/democratic"
    )


def test_a_primary_and_a_general_are_different_contests() -> None:
    """They have different candidates, polls and outcomes.

    Collapsing them is the modelling error the stage segment exists to prevent.
    """

    primary = ContestKey.build(
        division=MICHIGAN,
        office="governor",
        cycle=2026,
        stage=ContestStage.PRIMARY,
        party="democratic",
    )
    general = ContestKey.build(
        division=MICHIGAN,
        office="governor",
        cycle=2026,
        stage=ContestStage.GENERAL,
    )

    assert str(primary) != str(general)


def test_two_parties_primaries_are_different_contests() -> None:
    def primary_for(party: str) -> ContestKey:
        return ContestKey.build(
            division=MICHIGAN,
            office="governor",
            cycle=2026,
            stage=ContestStage.PRIMARY,
            party=party,
        )

    assert str(primary_for("democratic")) != str(primary_for("republican"))


def test_a_general_cannot_be_party_scoped() -> None:
    """A general is contested between parties, not within one.

    Allowing a party here would split one general into one contest per party,
    each with a subset of the candidates.
    """

    with pytest.raises(ValueError, match="cannot be party-scoped"):
        ContestKey.build(
            division=MICHIGAN,
            office="governor",
            cycle=2026,
            stage=ContestStage.GENERAL,
            party="democratic",
        )


def test_the_district_lives_in_the_division_not_the_office() -> None:
    """So a House contest joins to the jurisdiction the OCD importer created.

    Putting the district in the office instead would make the geography
    unmatchable against OCD, which is the only thing that resolves exactly.
    """

    key = ContestKey.build(
        division="ocd-division/country:us/state:mi/cd:11",
        office="US House",
        cycle=2026,
        stage=ContestStage.GENERAL,
    )
    assert str(key) == "ocd-division/country:us/state:mi/cd:11/us-house/2026/general"


@pytest.mark.parametrize(
    "key",
    [
        ContestKey(MICHIGAN, "governor", 2026, ContestStage.GENERAL),
        ContestKey(MICHIGAN, "governor", 2026, ContestStage.PRIMARY, "democratic"),
        ContestKey(
            "ocd-division/country:us/state:mi/cd:11",
            "us-house",
            2026,
            ContestStage.RUNOFF,
            "republican",
        ),
        ContestKey("ocd-division/country:us", "president", 2028, ContestStage.CAUCUS),
    ],
)
def test_a_key_round_trips(key: ContestKey) -> None:
    """Parsing has to survive the division containing slashes of its own."""

    assert ContestKey.parse(str(key)) == key


def test_a_malformed_key_is_refused_at_the_model_boundary() -> None:
    """A model may offer one, so the pattern has to reject nonsense.

    A key nothing else will ever derive is worse than no key: it mints a
    contest that looks identified and is unreachable.
    """

    with pytest.raises(ValueError):
        ScrapedEntity(name="Michigan Governor 2026", contest_key="michigan-governor")

    good = ScrapedEntity(
        name="Michigan Governor 2026",
        contest_key="ocd-division/country:us/state:mi/governor/2026/general",
    )
    mention = good.as_mention(EntityKind.CONTEST)
    assert [
        (identifier.namespace, identifier.value) for identifier in mention.identifiers
    ] == [(CONTEST_KEY_NAMESPACE, good.contest_key)]


def test_the_namespace_is_registered_and_loses_to_real_authorities() -> None:
    """Derived by us from other people's facts, so anyone else's ID should win."""

    specs = {spec.namespace: spec for spec in NAMESPACE_SPECS}
    contest_key = specs[CONTEST_KEY_NAMESPACE]
    assert contest_key.authority is None
    assert contest_key.precedence > specs["wikidata"].precedence
    assert contest_key.precedence > specs["ocd-division"].precedence


def test_an_implausible_cycle_is_a_parse_error_upstream() -> None:
    with pytest.raises(ValueError, match="plausible election cycle"):
        ContestKey.build(
            division=MICHIGAN, office="governor", cycle=26, stage=ContestStage.GENERAL
        )


def test_normalize_slug_refuses_something_with_no_content() -> None:
    assert normalize_slug("  U.S.  Senate ") == "u-s-senate"
    with pytest.raises(ValueError, match="no usable characters"):
        normalize_slug("!!!")
