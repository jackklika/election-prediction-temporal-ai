from typing import Sequence

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from predictelection.clients._base_config import ConfigBase
from temporalio.client import Client, Plugin


def _default_plugins() -> list[Plugin]:
    """PydanticAIPlugin installs the Pydantic data converter.

    Without it, workflow and activity arguments round-trip through plain JSON and
    come back as dicts rather than the contract models.
    """

    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

    return [PydanticAIPlugin()]


class TemporalConfig(ConfigBase):
    model_config = SettingsConfigDict(env_prefix="temporal_")

    host: str = Field(default="localhost:7233")
    namespace: str = Field(default="default")
    api_key: str | None = None
    task_queue: str = "default"


class TemporalClient:
    def __init__(self, *, config: TemporalConfig | None = None):
        self._config: TemporalConfig = TemporalConfig() if not config else config
        self._client: Client | None = None

    async def connect(self, *, plugins: Sequence[Plugin] | None = None):
        if self._client is not None:
            return
        if plugins is None:
            plugins = _default_plugins()
        self._client = await Client.connect(
            target_host=self._config.host,
            namespace=self._config.namespace,
            api_key=self._config.api_key,
            plugins=list(plugins),
        )

    @classmethod
    async def create(
        cls,
        *,
        config: TemporalConfig | None = None,
        plugins: Sequence[Plugin] | None = None,
    ) -> "TemporalClient":
        """Simple way to create and initialize the async client in one line"""
        self = cls(config=config)
        await self.connect(plugins=plugins)
        return self

    @property
    def client(self) -> Client:
        """
        Because the connection is async, it needs to be tied to an async runtime, so we need to do a two-step 1. create client 2. connect client.
        """
        if self._client is None:
            raise RuntimeError(
                "TemporalClient is not yet connected, call `await client.connect()` first"
            )
        return self._client

    @property
    def task_queue(self) -> str:
        return self._config.task_queue
