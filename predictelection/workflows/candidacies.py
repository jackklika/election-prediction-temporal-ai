"""Research who ran in a race, when, and who backed them.

Two record kinds come back from one pass, and `gather` returns them as one mixed
tuple. That works without any new machinery because `ScrapedPayload` is a tagged
union and the ingest activity dispatches per record through `INGESTORS` — which
is what the frozen seam was for. A domain that needed the loop changed to carry
two kinds would mean the seam had leaked.
"""

from __future__ import annotations

from temporalio import workflow

from predictelection.activities.contracts import (
    FindEntitiesInput,
    FindEntitiesOutput,
    ResearchInput,
    ResearchOutput,
)
from predictelection.research.scraped import ScrapedRecord
from predictelection.workflows.base import (
    READ_ACTIVITY,
    Activity,
    ResearchWorkflow,
)

with workflow.unsafe.imports_passed_through():
    from predictelection.agents.candidacies import candidacy_agent
    from predictelection.sql import EntityKind


KNOWN_PERSON_LIMIT = 50


# sandboxed=False: see the rationale in workflows/base.py.
@workflow.defn(sandboxed=False)
class ResearchCandidaciesWorkflow(ResearchWorkflow):
    task_type = "find_candidacies"

    agent = candidacy_agent
    __pydantic_ai_agents__ = [candidacy_agent]

    @workflow.run
    async def run(self, request: ResearchInput) -> ResearchOutput:
        # Temporal requires the run method on the class carrying @workflow.defn,
        # so this cannot simply be inherited. The body is in ResearchWorkflow.
        return await self.research(request)

    async def gather(self, request: ResearchInput) -> tuple[ScrapedRecord, ...]:
        known: FindEntitiesOutput = await workflow.execute_activity(
            Activity.FIND_ENTITIES,
            FindEntitiesInput(
                name=request.subject,
                kind=EntityKind.PERSON,
                limit=KNOWN_PERSON_LIMIT,
            ),
            result_type=FindEntitiesOutput,
            **READ_ACTIVITY,
        )
        if known.truncated:
            workflow.logger.warning(
                "known-person context truncated at %d for %r",
                KNOWN_PERSON_LIMIT,
                request.subject,
            )

        run = await self.agent.run(
            f"Reconstruct the candidacy timeline for {request.subject}: every "
            "candidate, the periods each was in the race, and the endorsements "
            "made during it — including any withdrawals and re-entries."
            + _already_recorded(known)
        )
        # One mixed tuple: the registry dispatches each record by its own type.
        return run.output.candidacies + run.output.endorsements


def _already_recorded(known: FindEntitiesOutput) -> str:
    """People already in the graph, so the agent reuses their spelling.

    People have no derived key — a person is a name plus whatever identifiers a
    source offered — so unlike contests, echoing an existing spelling is the only
    thing keeping "David Crowley" and "Dave Crowley" one entity.
    """

    if not known.matches:
        return ""
    lines = "\n".join(f"- {match.canonical_name}" for match in known.matches)
    return (
        "\n\nALREADY RECORDED — these people are in the graph. Still report any "
        f"you find; use these spellings exactly when you do:\n{lines}"
    )
