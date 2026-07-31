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
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from predictelection.sql import create_schema


DEFAULT_TEST_POSTGRES_URL = (
    "postgresql+psycopg://postgres:password@localhost:5432/predictelection"
)

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
    return os.environ.get("POSTGRES_URL", DEFAULT_TEST_POSTGRES_URL)


@pytest.fixture(scope="session")
def postgres_engine(pytestconfig: Config) -> Iterator[Engine]:
    """A session-wide engine against a freshly rebuilt schema."""

    required = pytestconfig.getoption("--require-postgres")
    url = _postgres_url()
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
            f"PostgreSQL is not reachable at {url}: {error.__class__.__name__}. "
            "Start it with `docker compose up -d`."
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
