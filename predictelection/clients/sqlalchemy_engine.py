from pydantic import model_validator
from pydantic_settings import SettingsConfigDict
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from predictelection.clients._base_config import ConfigBase


DEFAULT_POOL_SIZE = 20
"""Above the worker's activity thread count.

The activities are sync, so each concurrent one holds a connection for its whole
duration. SQLAlchemy's default pool is 5 + 10 overflow = 15, under
`worker.DEFAULT_ACTIVITY_THREADS` of 16 — so at full load the sixteenth activity
blocked on pool checkout rather than on the database. Sized here rather than
inherited so the two numbers are visibly related.
"""

DEFAULT_MAX_OVERFLOW = 10


class PostgresConfig(ConfigBase):
    model_config = SettingsConfigDict(env_prefix="postgres_")
    url: str
    echo: bool = False
    """Log every statement. Debugging only — it was on by default, which made
    the worker log the full text of every insert it ever ran."""

    pool_size: int = DEFAULT_POOL_SIZE
    max_overflow: int = DEFAULT_MAX_OVERFLOW

    @model_validator(mode="after")
    def _ensure_postgres_url(self):
        if not self.url.startswith("postgresql"):
            raise ValueError(
                "Connection string must start with postgresql -- only postgres supported"
            )
        if self.url.startswith("postgresql://"):
            # SQLAlchemy reads a bare postgresql:// as psycopg2, which this
            # project does not install; the dependency is psycopg 3. Normalizing
            # here keeps a plain connection string working everywhere.
            object.__setattr__(
                self,
                "url",
                self.url.replace("postgresql://", "postgresql+psycopg://", 1),
            )
        return self


class SqlAlchemyEngineClient:
    def __init__(self, config: PostgresConfig | None = None) -> None:
        self._config: PostgresConfig = config or PostgresConfig()  # ty: ignore[missing-argument]
        self.engine: Engine = create_engine(
            url=self._config.url,
            connect_args={
                "options": "-c timezone=utc"  # ensure we are always using utc
            },
            echo=self._config.echo,
            pool_size=self._config.pool_size,
            max_overflow=self._config.max_overflow,
            # A worker idles between runs, and Postgres or anything between it
            # and here may drop the connection meanwhile. Without this the next
            # activity fails on a stale socket and burns a retry to find out.
            pool_pre_ping=True,
        )
        self.session_factory: sessionmaker[Session] = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,  # We can return "dead" orm objects without triggering refreshes
        )


if __name__ == "__main__":
    SqlAlchemyEngineClient()
