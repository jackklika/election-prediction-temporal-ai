"""The seam that decides whether a new scrape type is additive.

Adding a domain is meant to be three lines in `research/registry.py` and nothing
else. These tests hold that shape: they fail if a record can reach the ingest
activity without an ingestor, if the wire contract stops round-tripping a
concrete record type, or if the generic layers grow a mention of a domain.

No Postgres needed — none of this touches the database.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
import inspect
from typing import Any
import uuid

import pytest

from predictelection.activities import contracts
from predictelection.activities.contracts import IngestRecordInput
from predictelection.research.debates import ScrapedDebate, ingest_debate
from predictelection.research.registry import (
    INGESTORS,
    ingestor_for,
    payload_types,
)
from predictelection.research.scraped import ScrapedEntity, ScrapedRecord
from predictelection.workflows import base as workflow_base


def _debate(**overrides: Any) -> ScrapedDebate:
    base: dict[str, Any] = {
        "title": "2026 Michigan Gubernatorial Debate",
        "source_url": "https://example.test/mi-debate",
        "starts_at": datetime(2026, 9, 15, 21, 0, tzinfo=UTC),
        "participants": (ScrapedEntity(name="Abdul El-Sayed"),),
    }
    return ScrapedDebate(**(base | overrides))


def test_every_payload_type_has_an_ingestor() -> None:
    """A half-registered domain would silently record nothing for its records.

    The union is what the activity accepts; INGESTORS is what it can act on. If
    they drift, a record arrives, dispatch raises inside a retry loop, and the
    run fails after it has already opened. Catching it here costs nothing.
    """

    assert set(payload_types()) == set(INGESTORS)


def test_every_ingestor_takes_record_then_context() -> None:
    """Uniform signature is what lets the registry dispatch without knowing."""

    for record_type, ingest in INGESTORS.items():
        parameters = list(inspect.signature(ingest).parameters)
        assert len(parameters) == 2, (
            f"{record_type.__name__}'s ingestor must take exactly (record, context)"
        )
        assert parameters[1] == "context"


def test_the_ingest_contract_round_trips_a_concrete_record() -> None:
    """Temporal serializes this through the Pydantic converter.

    The discriminator is what makes it work: `record: ScrapedRecord` would
    validate a debate against the base class and raise under extra="forbid",
    losing every field the domain added.
    """

    original = IngestRecordInput(record=_debate(), source_snapshot_id=uuid.uuid4())
    restored = IngestRecordInput.model_validate_json(original.model_dump_json())

    assert isinstance(restored.record, ScrapedDebate)
    assert restored.record == original.record
    assert restored.record.record_type == "debate"


def test_dispatch_selects_by_the_records_own_type() -> None:
    assert ingestor_for(_debate()) is ingest_debate


def test_an_unregistered_record_raises_rather_than_recording_nothing() -> None:
    class ScrapedNothing(ScrapedRecord):
        pass

    with pytest.raises(LookupError, match="not in INGESTORS"):
        ingestor_for(ScrapedNothing(source_url="https://example.test/x"))


def test_the_generic_layers_import_no_domain_module() -> None:
    """The acceptance test for the whole refactor.

    The contracts, the workflow scaffold and the activities are meant to be
    domain-free, so that adding a scrape type touches only its own module and
    the registry. Adding a domain used to mean editing all three plus the
    worker; if one of them imports a domain module again, that per-domain edit
    has crept back into a layer that was supposed to be finished.

    Imports rather than a word search: these modules legitimately *discuss*
    debates in prose explaining why they no longer depend on them.
    """

    from predictelection.activities import research as research_activities

    domain_modules = {ingest.__module__ for ingest in INGESTORS.values()}
    assert domain_modules, "no domains registered, so this would pass vacuously"

    for module in (contracts, workflow_base, research_activities):
        tree = ast.parse(inspect.getsource(module))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)

        leaked = imported & domain_modules
        assert not leaked, (
            f"{module.__name__} imports {leaked}; it is supposed to reach domains "
            "only through research.registry"
        )
