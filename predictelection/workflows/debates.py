"""Research a politician's debates and record them as attributed claims.

The division of labour is the point:

- the agent reads the web and reports what it found, and never writes;
- activities fetch, archive, and write, and never reason;
- this workflow only decides what happens next, so it can replay deterministically.

A debate whose source cannot be fetched is skipped rather than stored. Ingesting
it anyway would create claims citing a page nobody can check, which is the one
outcome the provenance model exists to prevent.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

from predictelection.activities.contracts import (
    ArchiveUrlInput,
    ArchiveUrlOutput,
    FinishResearchRunInput,
    FinishResearchRunOutput,
    IngestDebateInput,
    IngestDebateOutput,
    ResearchDebatesInput,
    ResearchDebatesOutput,
    StartResearchRunInput,
    StartResearchRunOutput,
)

with workflow.unsafe.imports_passed_through():
    from pydantic_ai.durable_exec.temporal import PydanticAIWorkflow

    from predictelection.agents.debates import DebateFindings, debate_agent
    from predictelection.sql import ResearchRunStatus, SourceKind


WRITE_ACTIVITY = workflow.ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=5),
)
FETCH_ACTIVITY = workflow.ActivityConfig(
    start_to_close_timeout=timedelta(seconds=90),
    retry_policy=RetryPolicy(maximum_attempts=3),
)


# sandboxed=False deliberately. Temporal's sandbox re-imports every module a
# workflow touches, and three things in this dependency tree defeat that on
# Python 3.14: beartype (pulled in by pydantic-ai via py-key-value-aio) installs
# a global import hook that hits a circular import under the sandbox importer,
# SQLAlchemy trips "__type_params__ must be set to a tuple" on re-import, and
# passing our own package through to avoid both leaves the workflow uninstrumented
# anyway — the same result with more machinery and less clarity.
#
# What the sandbox would protect against is absent here by construction: the body
# below only awaits activities and the agent. No clock, no randomness, no I/O, no
# mutable global state. Everything non-deterministic is already an activity,
# which is where it has to be regardless of this setting.
@workflow.defn(sandboxed=False)
class ResearchDebatesWorkflow(PydanticAIWorkflow):
    # Referenced through the class rather than the module global so a subclass can
    # substitute a stub agent. agent.override() cannot do it: the model call is an
    # activity, and the worker's task does not inherit the overriding context.
    agent = debate_agent
    __pydantic_ai_agents__ = [debate_agent]

    @workflow.run
    async def run(self, request: ResearchDebatesInput) -> ResearchDebatesOutput:
        started: StartResearchRunOutput = await workflow.execute_activity(
            "start_research_run",
            StartResearchRunInput(
                task_type="find_debates",
                subject=request.subject,
                workflow_id=workflow.info().workflow_id,
                workflow_run_id=workflow.info().run_id,
                agent_name=self.agent.name,
            ),
            result_type=StartResearchRunOutput,
            **WRITE_ACTIVITY,
        )

        try:
            findings = await self._find_debates(request.subject)
            result = await self._record(request, started, findings)
        except Exception as error:
            await workflow.execute_activity(
                "finish_research_run",
                FinishResearchRunInput(
                    research_run_id=started.research_run_id,
                    status=ResearchRunStatus.FAILED,
                    error_message=str(error)[:2000],
                ),
                result_type=FinishResearchRunOutput,
                **WRITE_ACTIVITY,
            )
            raise

        await workflow.execute_activity(
            "finish_research_run",
            FinishResearchRunInput(
                research_run_id=started.research_run_id,
                status=ResearchRunStatus.SUCCEEDED,
            ),
            result_type=FinishResearchRunOutput,
            **WRITE_ACTIVITY,
        )
        return result

    async def _find_debates(self, subject: str) -> DebateFindings:
        run = await self.agent.run(
            f"Find the notable debates {subject} has taken part in."
        )
        return run.output

    async def _record(
        self,
        request: ResearchDebatesInput,
        started: StartResearchRunOutput,
        findings: DebateFindings,
    ) -> ResearchDebatesOutput:
        event_ids: list = []
        skipped: list[str] = []
        new_debates = 0
        created = corroborated = unchanged = misaligned = 0

        for debate in findings.debates:
            try:
                archived: ArchiveUrlOutput = await workflow.execute_activity(
                    "archive_url",
                    ArchiveUrlInput(
                        url=debate.source_url,
                        kind=SourceKind.WEB_PAGE,
                        title=debate.title,
                    ),
                    result_type=ArchiveUrlOutput,
                    **FETCH_ACTIVITY,
                )
            except ActivityError:
                # An unfetchable citation is not a workflow failure; it means
                # this one debate cannot be evidenced, so it does not get stored.
                workflow.logger.warning("could not archive %s", debate.source_url)
                skipped.append(debate.source_url)
                continue

            ingested: IngestDebateOutput = await workflow.execute_activity(
                "ingest_debate",
                IngestDebateInput(
                    debate=debate,
                    source_snapshot_id=archived.source_snapshot_id,
                    research_run_id=started.research_run_id,
                    asserted_by=request.asserted_by or debate_agent.name,
                ),
                result_type=IngestDebateOutput,
                **WRITE_ACTIVITY,
            )
            event_ids.append(ingested.event_id)
            new_debates += 1 if ingested.event_created else 0
            created += ingested.claims_created
            corroborated += ingested.claims_corroborated
            unchanged += ingested.claims_unchanged
            misaligned += ingested.misaligned_count

        return ResearchDebatesOutput(
            research_run_id=started.research_run_id,
            debates_found=len(findings.debates),
            debates_new=new_debates,
            debates_already_known=len(event_ids) - new_debates,
            claims_created=created,
            claims_corroborated=corroborated,
            claims_unchanged=unchanged,
            misaligned_count=misaligned,
            event_ids=tuple(event_ids),
            skipped_urls=tuple(skipped),
        )
