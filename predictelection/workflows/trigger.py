"""Start a research workflow from the command line.

    make worker                                   # in one terminal
    make research SUBJECT="Abdul El-Sayed"        # in another

Only starts the workflow and waits for its result — all the work happens in the
worker, so the run survives this process exiting. Watch it at localhost:8080.
"""

from __future__ import annotations

import argparse
import asyncio

from predictelection.activities.contracts import (
    ResearchDebatesInput,
    ResearchDebatesOutput,
)
from predictelection.clients.temporal import TemporalClient


async def research_debates(subject: str, *, wait: bool = True) -> None:
    client = await TemporalClient.create()
    # Deterministic ID: starting the same research twice reattaches to the run in
    # flight rather than racing a second one against it.
    workflow_id = f"find-debates-{subject.strip().lower().replace(' ', '-')}"

    handle = await client.client.start_workflow(
        "ResearchDebatesWorkflow",
        ResearchDebatesInput(subject=subject),
        id=workflow_id,
        task_queue=client.task_queue,
        result_type=ResearchDebatesOutput,
    )
    print(f"started {handle.id} (run {handle.result_run_id})", flush=True)
    print("watch it at http://localhost:8080", flush=True)
    if not wait:
        return

    result = await handle.result()
    print()
    print(f"debates found       {result.debates_found}")
    print(f"  new to the graph  {result.debates_new}")
    print(f"  already known     {result.debates_already_known}")
    print(f"claims created      {result.claims_created}")
    print(f"claims corroborated {result.claims_corroborated}")
    if result.claims_unchanged:
        print(
            f"claims unchanged    {result.claims_unchanged}  (retry, nothing written)"
        )
    print(f"needing review      {result.misaligned_count}")
    for url in result.skipped_urls:
        print(f"skipped (uncitable) {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", help="Politician or race to research.")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start the run and exit instead of waiting for it.",
    )
    args = parser.parse_args()
    asyncio.run(research_debates(args.subject, wait=not args.no_wait))


if __name__ == "__main__":
    main()
