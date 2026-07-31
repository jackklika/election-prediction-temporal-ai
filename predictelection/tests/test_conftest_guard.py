"""The fixture layer's own safety check.

`postgres_engine` drops every table it owns. Pointed at the application's
database that is silent data loss — it deleted a live research run's rows
mid-flight, and the workflow then failed on `no research run <id>` while trying
to record its own failure.

The guard that prevents it compares two URLs written by different hands, so it
is exactly the kind of check that can pass while doing nothing. The first
version did: `.env` omits the port and the test URL spells out 5432, so the
tuples never matched and the suite ran happily against production data. These
tests exist because that failure is invisible from the outside.

No database needed — this is pure URL comparison.
"""

from __future__ import annotations

import pytest
from _pytest.outcomes import Failed
from sqlalchemy.engine import make_url

from predictelection.tests.conftest import (
    DEFAULT_TEST_POSTGRES_URL,
    TEST_URL_ENV,
    _refuse_the_application_database,
)


APP = "postgresql+psycopg://postgres:password@localhost/predictelection"


@pytest.fixture(autouse=True)
def _application_points_at_the_dev_database(monkeypatch: pytest.MonkeyPatch):
    """Pin the application's URL so these tests do not depend on a local .env."""

    monkeypatch.setenv("POSTGRES_URL", APP)


@pytest.mark.parametrize(
    "test_url",
    [
        pytest.param(APP, id="identical"),
        pytest.param(
            "postgresql+psycopg://postgres:password@localhost:5432/predictelection",
            id="explicit-default-port",
        ),
        pytest.param(
            "postgresql+psycopg://other:other@localhost/predictelection",
            id="different-credentials-same-database",
        ),
    ],
)
def test_the_application_database_is_refused(test_url: str) -> None:
    """Same server and database name means the same data, however it is spelled.

    The explicit-port case is the one that slipped through: it is the same
    server, written the other way round.
    """

    with pytest.raises(Failed, match="refusing to run"):
        _refuse_the_application_database(make_url(test_url))


@pytest.mark.parametrize(
    "test_url",
    [
        pytest.param(DEFAULT_TEST_POSTGRES_URL, id="the-default"),
        pytest.param(
            "postgresql+psycopg://postgres:password@localhost/something_else",
            id="another-database",
        ),
        pytest.param(
            "postgresql+psycopg://postgres:password@db.example:5432/predictelection",
            id="same-name-different-host",
        ),
    ],
)
def test_a_separate_database_is_allowed(test_url: str) -> None:
    _refuse_the_application_database(make_url(test_url))


def test_the_default_is_not_the_application_database() -> None:
    """The default must be safe on a machine with no configuration at all."""

    assert make_url(DEFAULT_TEST_POSTGRES_URL).database != make_url(APP).database


def test_the_suite_does_not_read_the_applications_variable() -> None:
    """Reading POSTGRES_URL was the other half of the bug.

    Pointing the application at a database would otherwise also point the
    table-dropping fixture at it.
    """

    assert TEST_URL_ENV != "POSTGRES_URL"


def test_a_missing_application_config_does_not_block_the_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Someone with no .env and no POSTGRES_URL should still be able to run tests."""

    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.setattr(
        "predictelection.clients._base_config._ENV_PATH", "/nonexistent/.env"
    )
    _refuse_the_application_database(make_url(DEFAULT_TEST_POSTGRES_URL))
