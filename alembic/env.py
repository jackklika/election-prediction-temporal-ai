"""Alembic environment.

The connection URL comes from `PostgresConfig`, not from alembic.ini, so a
migration cannot be run against a different database than the application reads
— which is the failure mode of keeping the URL in two places.

`create_all` remains the path for tests, where the database is empty and speed
matters. It is not a migration path: it cannot alter an existing CHECK
constraint or enum, so adding a predicate kind or a constraint branch to a live
database through it silently does nothing. Anything touching a database with
data in it goes through here.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from predictelection.clients.sqlalchemy_engine import PostgresConfig
from predictelection.sql.schema import AUTOGENERATE_OPTIONS


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set here rather than in alembic.ini: the URL is a deployment fact, and
# PostgresConfig already normalises postgresql:// to the psycopg 3 driver.
# `alembic -x url=...` overrides it, so a scratch database can be migrated
# without pointing the whole application at it.
config.set_main_option(
    "sqlalchemy.url",
    _url := (
        context.get_x_argument(as_dictionary=True).get("url") or PostgresConfig().url  # ty: ignore[missing-argument]
    ),
)

# AUTOGENERATE_OPTIONS carries target_metadata, the PostGIS filter and the
# type/server-default comparisons. It lives in predictelection.sql.schema so the
# migration tests can import the same settings this file runs with — importing
# it is also what registers all 32 tables, since a submodule that stopped being
# imported there would silently drop out of every future migration.


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting, for review before applying."""

    context.configure(
        url=_url,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **AUTOGENERATE_OPTIONS,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={"options": "-c timezone=utc"},
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, **AUTOGENERATE_OPTIONS)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
