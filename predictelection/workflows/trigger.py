"""Start a research workflow from the command line.

    make worker                                          # in one terminal
    make research SUBJECT="Abdul El-Sayed"               # in another
    make research SUBJECT="Michigan 2026" KIND=structure

Only starts the workflow and waits for its result — all the work happens in the
worker, so the run survives this process exiting. Watch it at localhost:8080.

Imports no workflow class, only its name: building one constructs its agent, and
starting a run should not require an API key.
"""

from __future__ import annotations

import argparse
import asyncio

from predictelection.activities.contracts import (
    ResearchInput,
    ResearchOutput,
)
from predictelection.clients.temporal import TemporalClient
from predictelection.workflows.names import (
    DEFAULT_RESEARCH_WORKFLOW,
    RESEARCH_WORKFLOWS,
)


async def research(
    subject: str,
    *,
    kind: str = DEFAULT_RESEARCH_WORKFLOW,
    wait: bool = True,
) -> ResearchOutput | None:
    client = await TemporalClient.create()
    # Deterministic ID: starting the same research twice reattaches to the run in
    # flight rather than racing a second one against it.
    slug = subject.strip().lower().replace(" ", "-")
    workflow_id = f"find-{kind}-{slug}"

    handle = await client.client.start_workflow(
        RESEARCH_WORKFLOWS[kind],
        ResearchInput(subject=subject),
        id=workflow_id,
        task_queue=client.task_queue,
        result_type=ResearchOutput,
    )
    print(f"started {handle.id} (run {handle.result_run_id})", flush=True)
    print("watch it at http://localhost:8080", flush=True)
    if not wait:
        return None

    result = await handle.result()
    print()
    print(f"records found       {result.records_found}")
    print(f"  new to the graph  {result.records_new}")
    print(f"  already known     {result.records_already_known}")
    print(f"claims created      {result.claims_created}")
    print(f"claims corroborated {result.claims_corroborated}")
    if result.claims_unchanged:
        print(
            f"claims unchanged    {result.claims_unchanged}  (retry, nothing written)"
        )
    print(f"needing review      {result.misaligned_count}")
    for url in result.skipped_urls:
        print(f"skipped (uncitable) {url}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", help="Politician or race to research.")
    parser.add_argument(
        "--kind",
        choices=sorted(RESEARCH_WORKFLOWS),
        default=DEFAULT_RESEARCH_WORKFLOW,
        help="Which research workflow to run.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start the run and exit instead of waiting for it.",
    )
    args = parser.parse_args()
    asyncio.run(research(args.subject, kind=args.kind, wait=not args.no_wait))


if __name__ == "__main__":
    main()
