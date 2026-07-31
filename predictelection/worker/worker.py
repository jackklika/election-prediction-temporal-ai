"""Worker runtime for temporal."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from temporalio.worker import Worker

from predictelection.clients.temporal import TemporalClient


DEFAULT_ACTIVITY_THREADS = 16
"""The activities are sync, so each concurrent one occupies a thread."""


class TemporalWorker:
    def __init__(
        self,
        *,
        client: TemporalClient,
        workflows: Sequence[Any],
        activities: Sequence[Any],
        activity_threads: int = DEFAULT_ACTIVITY_THREADS,
    ) -> None:
        self._client: TemporalClient = client  # needs to be connected
        # Sync activities need an executor; SQLAlchemy and boto3 both block, and
        # running them on the event loop would stall every other task here.
        self._executor = ThreadPoolExecutor(
            max_workers=activity_threads, thread_name_prefix="activity"
        )
        self._worker: Worker = Worker(
            self._client.client,
            task_queue=self._client.task_queue,
            workflows=list(workflows),
            activities=list(activities),
            activity_executor=self._executor,
        )

    @classmethod
    async def create(
        cls,
        *,
        client: TemporalClient | None = None,
        workflows: Sequence[Any] | None = None,
        activities: Sequence[Any] | None = None,
    ) -> "TemporalWorker":
        if not client:
            client = await TemporalClient.create()  # create + connects
        if workflows is None or activities is None:
            registered = _default_registrations()
            workflows = registered["workflows"] if workflows is None else workflows
            activities = registered["activities"] if activities is None else activities
        return TemporalWorker(client=client, workflows=workflows, activities=activities)

    @property
    def task_queue(self) -> str:
        return self._client.task_queue

    async def run(self) -> None:
        try:
            await self._worker.run()
        finally:
            self._executor.shutdown(wait=False)


def _default_registrations() -> dict[str, Sequence[Any]]:
    """Imported lazily: this reaches an LLM client, a database, and S3.

    Importing them at module scope would make `python -m predictelection.worker`
    the only thing that can import this file.
    """

    from predictelection.activities.research import build_activities
    from predictelection.workflows.debates import ResearchDebatesWorkflow

    return {
        "workflows": [ResearchDebatesWorkflow],
        "activities": build_activities().all(),
    }


def _configure_observability() -> None:
    """Set up logfire here rather than at import.

    The old agent module did this as an import side effect, which meant anything
    that merely referenced the agent also configured a global exporter. The
    process entrypoint is the only place that legitimately owns that decision.
    """

    import logfire
    from pydantic_ai import Agent

    logfire.configure()
    logfire.instrument_pydantic_ai()
    Agent.instrument_all()


if __name__ == "__main__":

    async def _main() -> None:
        _configure_observability()
        worker = await TemporalWorker.create()
        # flush: stdout is block-buffered when this runs under make or a pipe, and a
        # long-lived process would otherwise never show its banner.
        print(
            f"worker polling task queue {worker.task_queue!r} — ctrl-c to stop",
            flush=True,
        )
        await worker.run()

    asyncio.run(_main())
