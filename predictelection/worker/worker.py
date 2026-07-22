"""Worker runtime for temporal"""

import asyncio
from typing import Any, Sequence

from predictelection.clients.temporal import TemporalClient

from temporalio.worker import Worker


class TemporalWorker:
    def __init__(self, *, client: TemporalClient) -> None:
        self._client: TemporalClient = client  # needs to be connected
        self._worker: Worker = Worker(
            self._client.client,
            task_queue=self._client.task_queue,
            workflows=self.workflows,
            activities=self.activities,
        )

    @classmethod
    async def create(
        cls, *, client: TemporalClient | None = None
    ) -> "TemporalWorker":
        if not client:
            client = await TemporalClient.create()  # create + connects
        return TemporalWorker(client=client)

    @property
    def workflows(self) -> Sequence[Any]:
        return []

    @property
    def activities(self) -> Sequence[Any]:
        return []

    async def run(self) -> None:
        await self._worker.run()


if __name__ == "__main__":
    import asyncio

    async def _main():
        worker = await TemporalWorker.create()
        await worker.run()

    asyncio.run(_main())
