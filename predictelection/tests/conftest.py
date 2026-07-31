"""Database fixtures for the tests that need a real PostgreSQL.

Most of this schema's guarantees are CHECK constraints, partial unique indexes,
NULLS NOT DISTINCT keys, and PostgreSQL regex predicates. None of those run
against SQLite, and compiling the DDL only proves it parses, so the tests that
matter need a live server.

By default a missing server skips those tests, which keeps `pytest` usable with
no Docker. Pass --require-postgres (see `make test-db`) to turn the skip into a
failure, so CI cannot quietly stop exercising the constraints.
"""

from __future__ import annotations

from collections.abc import Iterator
import os

from _pytest.config import Config
from _pytest.config.argparsing import Parser
import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from predictelection.sql import create_schema


DEFAULT_TEST_POSTGRES_URL = (
    "postgresql+psycopg://postgres:password@localhost:5432/predictelection_test"
)
"""A database of the suite's own, created on demand.

`postgres_engine` drops every table it owns before each session, so it must
never point at a database anything else is using. It previously defaulted to
`predictelection` — the application's own database — which meant running the
suite while a research run was in flight deleted that run's rows underneath it.
The workflow then failed on `no research run <id>` while trying to record its own
failure, which is a confusing way to discover you destroyed the data.
"""

TEST_URL_ENV = "TEST_POSTGRES_URL"
"""Deliberately *not* POSTGRES_URL.

Reading the application's variable was the other half of the same bug: pointing
the app at a database also pointed the table-dropping fixture at it. The suite
gets its own variable so the two cannot be aimed at one target by accident.
"""

_SUITE_OWNED_TABLES = text(
    """
    SELECT c.relname
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    LEFT JOIN pg_depend d ON d.objid = c.oid AND d.deptype = 'e'
    WHERE n.nspname = 'public' AND c.relkind = 'r' AND d.objid IS NULL
    """
)


def _drop_suite_tables(engine: Engine) -> None:
    """Drop every table this suite owns, leaving extension tables alone.

    Base.metadata.drop_all is not enough: a table from an older revision of the
    models is no longer in the metadata, so it survives, and its foreign keys
    then block dropping the tables that are still mapped. Dropping the whole
    public schema is too much in the other direction, because PostGIS keeps
    spatial_ref_sys there and CASCADE would take the extension with it.
    """

    with engine.begin() as connection:
        names = connection.execute(_SUITE_OWNED_TABLES).scalars().all()
        for name in names:
            connection.execute(text(f'DROP TABLE IF EXISTS public."{name}" CASCADE'))


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--require-postgres",
        action="store_true",
        default=False,
        help="Fail instead of skipping when PostgreSQL or MinIO is unreachable.",
    )


@pytest.fixture(scope="session")
def object_store(pytestconfig: Config):
    """A bucket on the local MinIO, which speaks the same API as S3.

    Testing against MinIO rather than a mock means the integration tests exercise
    the same S3ObjectStore that will talk to AWS, so a credential or addressing
    mistake surfaces here instead of in production.
    """

    from botocore.exceptions import BotoCoreError, ClientError

    from predictelection.storage import S3ObjectStore, local_minio_config

    store = S3ObjectStore(local_minio_config(bucket="predictelection-test"))
    try:
        store.ensure_bucket()
    except (ClientError, BotoCoreError, OSError) as error:
        message = (
            f"MinIO is not reachable: {error.__class__.__name__}. "
            "Start it with `docker compose up -d`."
        )
        if pytestconfig.getoption("--require-postgres"):
            pytest.fail(message, pytrace=False)
        pytest.skip(message, allow_module_level=True)
    return store


def _postgres_url() -> str:
    return os.environ.get(TEST_URL_ENV) or DEFAULT_TEST_POSTGRES_URL


def _refuse_the_application_database(url: URL) -> None:
    """Stop the suite before it drops tables in a database someone else owns.

    Belt to the separate-variable braces: `TEST_POSTGRES_URL` could still be
    pointed at the application's database by hand, and the failure mode is
    silent data loss rather than an error. Compared on host and database name
    because that is what identifies the target — the credentials in the two URLs
    need not match.
    """

    try:
        from predictelection.clients.sqlalchemy_engine import PostgresConfig

        application = make_url(PostgresConfig().url)  # ty: ignore[missing-argument]
    except Exception:  # noqa: BLE001 - no application config here is fine
        return

    def target(candidate: URL) -> tuple[str, int, str | None]:
        # Normalized, because the two URLs are written by different hands: a
        # connection string that omits the port is the same server as one that
        # spells out 5432, and comparing them raw let this guard pass while
        # pointed straight at the application's database.
        return (
            candidate.host or "localhost",
            candidate.port or 5432,
            candidate.database,
        )

    if target(url) == target(application):
        pytest.fail(
            f"refusing to run: the test database ({url.database}) is the "
            "application's own. This fixture drops every table it owns, so "
            "running here destroys real data — including any research run in "
            f"flight. Unset {TEST_URL_ENV} or point it somewhere else.",
            pytrace=False,
        )


def _ensure_database(url: URL, *, required: bool) -> bool:
    """Create the test database if it does not exist yet.

    Compose only provisions the application's database, so the suite makes its
    own rather than requiring a setup step nobody will remember.
    """

    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": url.database},
            ).scalar()
            if not exists:
                connection.execute(text(f'CREATE DATABASE "{url.database}"'))
    except OperationalError as error:
        message = (
            f"PostgreSQL is not reachable at {url.render_as_string()}: "
            f"{error.__class__.__name__}. Start it with `docker compose up -d`."
        )
        if required:
            pytest.fail(message, pytrace=False)
        pytest.skip(message, allow_module_level=True)
        return False
    finally:
        admin.dispose()
    return True


@pytest.fixture(scope="session")
def postgres_engine(pytestconfig: Config) -> Iterator[Engine]:
    """A session-wide engine against a freshly rebuilt schema."""

    required = pytestconfig.getoption("--require-postgres")
    url = make_url(_postgres_url())
    _refuse_the_application_database(url)
    _ensure_database(url, required=required)

    engine = create_engine(
        url,
        connect_args={"options": "-c timezone=utc"},
        poolclass=None,
    )

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except OperationalError as error:
        engine.dispose()
        message = (
            f"PostgreSQL is not reachable at {url.render_as_string()}: "
            f"{error.__class__.__name__}. Start it with `docker compose up -d`."
        )
        if required:
            pytest.fail(message, pytrace=False)
        pytest.skip(message, allow_module_level=True)

    # Rebuild rather than create: a leftover schema from an earlier model
    # version would mask real DDL breakage.
    _drop_suite_tables(engine)
    create_schema(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def session(postgres_engine: Engine) -> Iterator[Session]:
    """A session whose writes are always rolled back.

    The outer transaction never commits, so tests see the seeded predicate
    catalog and each other's absence. join_transaction_mode="create_savepoint"
    lets a test call session.commit() and still be undone here.
    """

    connection = postgres_engine.connect()
    transaction = connection.begin()
    test_session = Session(
        bind=connection,
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
    )
    try:
        yield test_session
    finally:
        test_session.close()
        transaction.rollback()
        connection.close()
