"""Research who funded a race.

Three declarations and a `gather`, like the others. Everything that is not about
donations — the run lifecycle, archiving each citation, ingest, counting, and
skipping a record whose source cannot be fetched — comes from
`workflows/base.py`, and none of it had to be touched to add this domain.
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
    from predictelection.agents.donations import DonationFindings, donation_agent
    from predictelection.sql import EntityKind


KNOWN_DONOR_LIMIT = 50


# sandboxed=False: see the rationale in workflows/base.py.
@workflow.defn(sandboxed=False)
class ResearchDonationsWorkflow(ResearchWorkflow):
    task_type = "find_donations"

    agent = donation_agent
    __pydantic_ai_agents__ = [donation_agent]

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
                kind=EntityKind.ORGANIZATION,
                limit=KNOWN_DONOR_LIMIT,
            ),
            result_type=FindEntitiesOutput,
            **READ_ACTIVITY,
        )
        if known.truncated:
            workflow.logger.warning(
                "known-donor context truncated at %d for %r",
                KNOWN_DONOR_LIMIT,
                request.subject,
            )

        findings: DonationFindings = await self.ask(
            f"Report the money raised and spent in {request.subject}: "
            "contributions to the candidates and their committees, and "
            "independent expenditures for or against them." + _already_recorded(known)
        )
        return findings.donations


def _already_recorded(known: FindEntitiesOutput) -> str:
    """Committees and PACs already in the graph, so the agent reuses spellings.

    Organizations, not people, because that is where the duplicates come from:
    a committee has a long legal name and a short reported one, and nothing
    derives a key for it the way `ContestKey` does for contests. Echoing an
    existing spelling is the only thing keeping "Fair Wisconsin PAC" and "Fair
    Wisconsin Political Action Committee" one entity — and if it fails, the
    review queue's merge is the repair.
    """

    if not known.matches:
        return ""
    lines = "\n".join(f"- {match.canonical_name}" for match in known.matches)
    return (
        "\n\nALREADY RECORDED — these organizations are in the graph. Still "
        f"report any you find; use these spellings exactly when you do:\n{lines}"
    )
