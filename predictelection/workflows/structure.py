"""Research the structure of the races a subject is involved in.

The whole workflow is below, and it is three declarations and a `gather`. Run
lifecycle, archiving, ingest, counting and skip-on-unfetchable all come from
`workflows/base.py` — if any of that had to be restated here, the seam would
have leaked.
"""

from __future__ import annotations

from temporalio import workflow

from predictelection.activities.contracts import (
    FindEntitiesInput,
    FindEntitiesOutput,
    ResearchInput,
    ResearchOutput,
)
from predictelection.workflows.base import (
    READ_ACTIVITY,
    Activity,
    ResearchWorkflow,
)

with workflow.unsafe.imports_passed_through():
    from predictelection.agents.structure import StructureFindings, structure_agent
    from predictelection.research.structure import ScrapedRaceStructure
    from predictelection.sql import EntityKind


KNOWN_CONTEST_LIMIT = 50


# sandboxed=False: see the rationale in workflows/base.py.
@workflow.defn(sandboxed=False)
class ResearchStructureWorkflow(ResearchWorkflow):
    task_type = "find_race_structure"

    agent = structure_agent
    __pydantic_ai_agents__ = [structure_agent]

    @workflow.run
    async def run(self, request: ResearchInput) -> ResearchOutput:
        # Temporal requires the run method on the class carrying @workflow.defn,
        # so this cannot simply be inherited. The body is in ResearchWorkflow.
        return await self.research(request)

    async def gather(self, request: ResearchInput) -> tuple[ScrapedRaceStructure, ...]:
        known: FindEntitiesOutput = await workflow.execute_activity(
            Activity.FIND_ENTITIES,
            FindEntitiesInput(
                name=request.subject,
                kind=EntityKind.CONTEST,
                limit=KNOWN_CONTEST_LIMIT,
            ),
            result_type=FindEntitiesOutput,
            **READ_ACTIVITY,
        )
        if known.truncated:
            workflow.logger.warning(
                "known-contest context truncated at %d for %r",
                KNOWN_CONTEST_LIMIT,
                request.subject,
            )

        findings: StructureFindings = await self.ask(
            "Describe the elections and contests for "
            f"{request.subject}, including both primaries and generals."
            + _already_recorded(known)
        )
        return findings.contests


def _already_recorded(known: FindEntitiesOutput) -> str:
    """Contests already in the graph.

    Less load-bearing than for debates, because a contest resolves by a derived
    key rather than by its name — two runs reach the same entity even if they
    phrase it differently. Still worth showing: it stops the agent re-reporting
    what is already there and spending a round trip on it.
    """

    if not known.matches:
        return ""
    lines = "\n".join(f"- {match.canonical_name}" for match in known.matches)
    return f"\n\nALREADY RECORDED:\n{lines}"
