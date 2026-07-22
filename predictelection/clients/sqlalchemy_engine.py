from pydantic import model_validator
from pydantic_settings import SettingsConfigDict
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from predictelection.clients._base_config import ConfigBase


class PostgresConfig(ConfigBase):
    model_config = SettingsConfigDict(env_prefix="postgres_")
    url: str

    @model_validator(mode="after")
    def _ensure_postgres_url(self):
        if not self.url.startswith("postgresql"):
            raise ValueError(
                "Connection string must start with postgresql -- only postgres supported"
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
            echo=True,
        )
        self.session_factory: sessionmaker[Session] = sessionmaker(
            engine=self.engine,
            expire_on_commit=False,  # We can return "dead" orm objects without triggering refreshes
        )


if __name__ == "__main__":
    SqlAlchemyEngineClient()
