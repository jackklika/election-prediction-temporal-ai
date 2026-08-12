"""Research a politician's debates and record them as attributed claims.

Everything about running a research workflow — opening the run, archiving each
citation, ingesting, counting, closing — lives in `workflows/base.py`. What is
left here is the part that is actually about debates: which agent to ask, and
what to show it so it stops re-describing debates it has already reported.
"""

from __future__ import annotations

from temporalio import workflow

from predictelection.activities.contracts import (
    FindEntitiesInput,
    FindEntitiesOutput,
    FindEventsInput,
    ResearchInput,
    ResearchOutput,
)
from predictelection.workflows.base import (
    READ_ACTIVITY,
    Activity,
    ResearchWorkflow,
)

with workflow.unsafe.imports_passed_through():
    from predictelection.agents.debates import debate_agent
    from predictelection.research.debates import ScrapedDebate
    from predictelection.sql import EntityKind, PoliticalEventKind


DEDUP_CONTEXT_LIMIT = 50
"""How many already-recorded debates to show the agent.

A cap is unavoidable — the prompt has a budget — so what matters is that the
rows it keeps are the subject's most recent ones and that hitting it is logged.
"""

SUBJECT_MATCH_LIMIT = 5
"""People one subject name may resolve to. All of their debates get shown."""


# sandboxed=False: see the rationale in workflows/base.py.
@workflow.defn(sandboxed=False)
class ResearchDebatesWorkflow(ResearchWorkflow):
    task_type = "find_debates"

    # Referenced through the class rather than the module global so a subclass can
    # substitute a stub agent. agent.override() cannot do it: the model call is an
    # activity, and the worker's task does not inherit the overriding context.
    agent = debate_agent
    __pydantic_ai_agents__ = [debate_agent]

    @workflow.run
    async def run(self, request: ResearchInput) -> ResearchOutput:
        # Temporal requires the run method on the class carrying @workflow.defn,
        # so this cannot simply be inherited. The body is in ResearchWorkflow.
        return await self.research(request)

    async def gather(self, request: ResearchInput) -> tuple[ScrapedDebate, ...]:
        """Look up what is already recorded, then ask the agent.

        The workflow fetches the context rather than the agent reaching for it.
        A tool cannot: TemporalDurability runs each agent step *as an activity*,
        and an activity cannot call execute_activity — wiring the lookup as a
        tool hangs. Fetching here also matches the classic construction pipeline,
        which links entities before extracting relations, and it keeps the agent
        purely a reader.
        """

        run = await self.agent.run(
            f"Find the notable debates {request.subject} has taken part in."
            + await self._already_recorded(request.subject)
        )
        return run.output.debates

    async def _already_recorded(self, subject: str) -> str:
        """The subject's own debates, not the alphabetically first 50 events.

        This used to ask for `kind=EVENT, limit=50` with no other filter, and
        `find_events` ordered by canonical_name — so past 50 events the agent was
        shown the alphabetically first 50, none of them necessarily the
        subject's, while this block still rendered as though it were complete.
        Scoping to the subject is what makes the list mean what it says.
        """

        people: FindEntitiesOutput = await workflow.execute_activity(
            Activity.FIND_ENTITIES,
            FindEntitiesInput(
                name=subject, kind=EntityKind.PERSON, limit=SUBJECT_MATCH_LIMIT
            ),
            result_type=FindEntitiesOutput,
            **READ_ACTIVITY,
        )
        if not people.matches:
            # Nothing about this subject is recorded yet, so there is nothing to
            # reuse. An empty block is honest; a global list would not be.
            return ""

        known: FindEntitiesOutput = await workflow.execute_activity(
            Activity.FIND_EVENTS,
            FindEventsInput(
                participant_ids=tuple(match.entity_id for match in people.matches),
                event_kind=PoliticalEventKind.DEBATE,
                limit=DEDUP_CONTEXT_LIMIT,
            ),
            result_type=FindEntitiesOutput,
            **READ_ACTIVITY,
        )
        if known.truncated:
            workflow.logger.warning(
                "dedup context truncated at %d debates for %r; "
                "the agent cannot see the rest and may duplicate them",
                DEDUP_CONTEXT_LIMIT,
                subject,
            )
        return _already_recorded(known)


def _already_recorded(known: FindEntitiesOutput) -> str:
    """Render existing events so the agent can echo a title back verbatim."""

    if not known.matches:
        return ""
    lines = "\n".join(
        f"- {match.canonical_name}"
        + (f"  ({match.occurred_at:%Y-%m-%d})" if match.occurred_at else "")
        for match in known.matches
    )
    # Saying the list is partial matters: presented as complete, it reads as
    # "everything else is new", which is the licence to duplicate.
    caveat = (
        "\n(This list is capped and shows the most recent ones only — older "
        "debates may already be recorded under titles not shown here.)"
        if known.truncated
        else ""
    )
    return (
        "\n\nALREADY RECORDED — these debates are in the graph already. Still "
        "report any of them you find; use the title exactly as written here "
        f"when you do:\n{lines}{caveat}"
    )
