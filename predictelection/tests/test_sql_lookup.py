"""Graph lookup — what the agent reads before it writes.

Two runs on one subject produced 11 event entities for 6 real debates, because
the agent re-described each one instead of reusing what it had already written.
The classic construction pipeline links entities *before* extracting relations;
these functions are what make that order possible here.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from predictelection.sql import (
    EntityKind,
    EntityMention,
    TimePrecision,
    Validity,
    find_entities,
    find_events,
    get_or_create_claim,
    get_predicate_spec,
    resolve_entity_mention,
)
from predictelection.tests import factories as f


pytestmark = pytest.mark.postgres


def _debate(session: Session, title: str, when: datetime):
    event = resolve_entity_mention(
        session, EntityMention(kind=EntityKind.EVENT, name=title)
    )
    get_or_create_claim(
        session,
        predicate=get_predicate_spec("event_occurrence"),
        subject_id=event.entity_id,
        value={"status": "occurred"},
        validity=Validity.between(when, None, TimePrecision.DAY),
    )
    session.flush()
    return event


def test_lookup_returns_the_stored_title_to_echo_back(session: Session) -> None:
    """The whole mechanism: hand the agent the name it already used."""

    stored = "Michigan Democratic Gubernatorial Primary Debate (WOOD TV8, Grand Rapids)"
    _debate(session, stored, datetime(2026, 6, 20, tzinfo=UTC))

    matches = find_events(session, name="Gubernatorial Primary Debate")
    assert [m.canonical_name for m in matches] == [stored]
    assert matches[0].occurred_at == datetime(2026, 6, 20, tzinfo=UTC)


def test_date_separates_debates_that_names_cannot(session: Session) -> None:
    """Two real debates a month apart share nearly every word.

    Name similarity alone would merge them. The date is what discriminates, and
    it is why the tool asks for a window rather than just a title.
    """

    _debate(
        session,
        "Michigan Democratic Gubernatorial Primary Debate (WOOD TV8, Grand Rapids)",
        datetime(2026, 6, 20, tzinfo=UTC),
    )
    _debate(
        session,
        "Michigan Democratic Gubernatorial Primary Debate (WDIV, Detroit)",
        datetime(2026, 7, 19, tzinfo=UTC),
    )

    june = find_events(
        session,
        occurred_after=datetime(2026, 6, 1, tzinfo=UTC),
        occurred_before=datetime(2026, 6, 30, tzinfo=UTC),
    )
    assert len(june) == 1
    assert "WOOD TV8" in june[0].canonical_name

    both = find_events(
        session,
        occurred_after=datetime(2026, 1, 1, tzinfo=UTC),
        occurred_before=datetime(2026, 12, 31, tzinfo=UTC),
    )
    assert len(both) == 2


def test_a_reworded_query_still_finds_the_existing_event(session: Session) -> None:
    """The rephrasing that caused the fork should now find the original."""

    stored = "Mackinac Policy Conference 2018 Gubernatorial Debate"
    _debate(session, stored, datetime(2018, 5, 31, tzinfo=UTC))

    # what the second run actually invented, searched by its date
    matches = find_events(
        session,
        occurred_after=datetime(2018, 5, 30, tzinfo=UTC),
        occurred_before=datetime(2018, 6, 1, tzinfo=UTC),
    )
    assert [m.canonical_name for m in matches] == [stored]


def test_find_entities_matches_on_aliases_too(session: Session) -> None:
    resolve_entity_mention(
        session,
        EntityMention(
            kind=EntityKind.PERSON,
            name="Abdul El-Sayed",
            aliases=("Dr. Abdul El-Sayed",),
        ),
    )
    session.flush()

    by_alias = find_entities(session, name="Dr. Abdul", kind=EntityKind.PERSON)
    assert [m.canonical_name for m in by_alias] == ["Abdul El-Sayed"]
    assert "Dr. Abdul El-Sayed" in by_alias[0].aliases


def test_lookup_is_scoped_by_kind(session: Session) -> None:
    resolve_entity_mention(
        session, EntityMention(kind=EntityKind.PERSON, name="Washington")
    )
    resolve_entity_mention(
        session, EntityMention(kind=EntityKind.JURISDICTION, name="Washington")
    )
    session.flush()

    people = find_entities(session, name="Washington", kind=EntityKind.PERSON)
    assert len(people) == 1
    assert people[0].kind is EntityKind.PERSON


def test_lookup_writes_nothing(session: Session) -> None:
    """Read-only, so an agent may call it freely and a retry cannot corrupt."""

    f.make_entity(session, kind=EntityKind.EVENT, canonical_name="A debate")
    session.flush()
    before = len(find_entities(session))

    for _ in range(3):
        find_entities(session, name="debate")
        find_events(session, name="debate")

    assert len(find_entities(session)) == before
