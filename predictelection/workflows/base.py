"""The scaffold every agent-driven research workflow runs on.

The division of labour is the point:

- the agent reads the web and reports what it found, and never writes;
- activities fetch, archive, and write, and never reason;
- the workflow only decides what happens next, so it can replay deterministically.

What is here rather than in each domain: opening and closing the research run,
archiving each record's citation, dispatching the ingest, counting outcomes, and
deciding that an unfetchable citation skips one record rather than failing the
run. None of that varies by domain, and all of it was previously retyped —
which is how two workflows end up disagreeing about whether a failed fetch is
fatal.

A domain workflow supplies three things: a `task_type`, an `agent`, and a
`gather` that returns records. Everything else it inherits.

A record whose source cannot be fetched is skipped rather than stored. Ingesting
it anyway would create claims citing a page nobody can check, which is the one
outcome the provenance model exists to prevent.

## sandboxed=False

Concrete workflows below are declared `@workflow.defn(sandboxed=False)`
deliberately. Temporal's sandbox re-imports every module a workflow touches, and
three things in this dependency tree defeat that on Python 3.14: beartype
(pulled in by pydantic-ai via py-key-value-aio) installs a global import hook
that hits a circular import under the sandbox importer, SQLAlchemy trips
"__type_params__ must be set to a tuple" on re-import, and passing our own
package through to avoid both leaves the workflow uninstrumented anyway — the
same result with more machinery and less clarity.

What the sandbox would protect against is absent here by construction: the body
below only awaits activities and the agent. No clock, no randomness, no I/O, no
mutable global state. Everything non-deterministic is already an activity, which
is where it has to be regardless of this setting.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

from predictelection.activities.contracts import (
    ArchiveUrlInput,
    ArchiveUrlOutput,
    FinishResearchRunInput,
    FinishResearchRunOutput,
    IngestRecordInput,
    IngestRecordOutput,
    ResearchInput,
    ResearchOutput,
    StartResearchRunInput,
    StartResearchRunOutput,
)

with workflow.unsafe.imports_passed_through():
    from pydantic_ai import Agent
    from pydantic_ai.durable_exec.temporal import PydanticAIWorkflow
    from pydantic_ai.messages import ModelResponse

    from predictelection.research.scraped import ScrapedRecord
    from predictelection.sql import ResearchRunStatus, SourceKind


WRITE_ACTIVITY = workflow.ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=5),
)
READ_ACTIVITY = workflow.ActivityConfig(
    start_to_close_timeout=timedelta(seconds=30),
    retry_policy=RetryPolicy(maximum_attempts=3),
)
FETCH_ACTIVITY = workflow.ActivityConfig(
    start_to_close_timeout=timedelta(seconds=90),
    retry_policy=RetryPolicy(maximum_attempts=3),
)


UNFINISHED_REASONS = frozenset({"length", "content_filter", "error"})
"""Finish reasons that mean the model stopped before it had said everything.

`stop` and `tool_call` are the two ways a response ends on purpose. Everything
else is the model being cut off, and on a structured output that arrives as a
*valid* object rather than as an error: every `*Findings` model defaults its
list fields to `()`, so a response truncated mid-generation parses as an honest
"found nothing".

That is not hypothetical. The candidacy agent's first two live runs both ended
with `finish_reason: length`, and both were reported as clean runs that found
zero candidates; the truncation was visible only in the Logfire trace. See
`AGENT_MAX_TOKENS` in agents/base.py for the other half of the fix — that raises
the ceiling, this makes hitting it audible.
"""


class Activity:
    """Activity names, which Temporal dispatches on as bare strings.

    Constants because a rename that misses a call site otherwise fails at
    runtime, inside a retry loop, after the run has already started writing.
    """

    START_RESEARCH_RUN = "start_research_run"
    FINISH_RESEARCH_RUN = "finish_research_run"
    ARCHIVE_URL = "archive_url"
    INGEST_RECORD = "ingest_record"
    FIND_ENTITIES = "find_entities"
    FIND_EVENTS = "find_events"


class ResearchWorkflow(PydanticAIWorkflow):
    """Run lifecycle, archiving and ingest, shared by every research domain.

    Not itself `@workflow.defn`, and `research` is deliberately not decorated
    `@workflow.run`. Temporal insists the run method be defined on the class
    carrying `@workflow.defn` — inheriting it raises "@workflow.run defined on
    ResearchWorkflow.run but not on the override". So each concrete workflow
    declares its own two-line entry point:

        @workflow.run
        async def run(self, request: ResearchInput) -> ResearchOutput:
            return await self.research(request)

    which keeps Temporal's registration explicit per workflow while the body it
    runs stays in one place.
    """

    task_type: ClassVar[str]
    """Names this kind of research on every ResearchRun row it opens."""

    agent: ClassVar[Agent[None, Any]]
    """Set by each domain. Read through the class, never as a module global, so
    a subclass that swaps in a stub does not attribute its claims to the agent
    it replaced."""

    async def gather(self, request: ResearchInput) -> tuple[ScrapedRecord, ...]:
        """Ask the agent what it can find. The one part a domain must write."""

        raise NotImplementedError

    async def ask(self, prompt: str) -> Any:
        """Run the domain agent, refusing an answer the model did not finish.

        Every `gather` goes through this rather than calling `self.agent.run`
        directly, because the failure it catches is invisible at the call site:
        a truncated structured response deserialises into a findings object with
        empty lists, which is indistinguishable from a run that genuinely found
        nothing. Four such runs were read as working ones in a single session.

        Non-retryable on purpose. An overrun is a budget problem, not a flake —
        a high-effort run that spent `AGENT_MAX_TOKENS` on thinking will spend it
        again — and `TemporalDurability` already retries the model call itself,
        so a retryable error here would multiply that. Raising instead lets
        `research` below record the run as FAILED with the reason attached.

        The returned output stays `Any`: `agent` is declared
        `Agent[None, Any]`, so each domain annotates the findings type at its own
        call site, where the agent that produces it is known.

        Kept deliberately dull, because this runs in workflow code: an
        `ApplicationError` fails the workflow, but any *other* exception fails the
        workflow *task*, which Temporal then retries forever. A typo in this
        method is therefore an infinite loop rather than a test failure — one
        already cost a hung test run here.
        """

        result = await self.agent.run(prompt)
        responses = [
            message
            for message in result.all_messages()
            if isinstance(message, ModelResponse)
        ]

        # An earlier truncated response is only a warning: pydantic-ai may
        # legitimately continue past one, and the answer that matters is the last.
        for response in responses[:-1]:
            if response.finish_reason in UNFINISHED_REASONS:
                workflow.logger.warning(
                    "%s: an intermediate response ended with %r",
                    self.agent.name,
                    response.finish_reason,
                )

        final = responses[-1] if responses else None
        if final is not None and (
            final.finish_reason in UNFINISHED_REASONS or final.state == "incomplete"
        ):
            raise ApplicationError(
                f"{self.agent.name} did not finish its answer: "
                f"finish_reason={final.finish_reason!r} state={final.state!r} "
                f"after {result.usage.output_tokens} output tokens. Whatever it "
                "did emit is incomplete, so it is not being stored. If the reason "
                "is 'length', raise AGENT_MAX_TOKENS in agents/base.py.",
                non_retryable=True,
            )

        return result.output

    async def research(self, request: ResearchInput) -> ResearchOutput:
        started: StartResearchRunOutput = await workflow.execute_activity(
            Activity.START_RESEARCH_RUN,
            StartResearchRunInput(
                task_type=self.task_type,
                subject=request.subject,
                workflow_id=workflow.info().workflow_id,
                workflow_run_id=workflow.info().run_id,
                agent_name=self.agent.name,
            ),
            result_type=StartResearchRunOutput,
            **WRITE_ACTIVITY,
        )

        try:
            records = await self.gather(request)
            result = await self._record(request, started, records)
        except Exception as error:
            await self._finish(started, ResearchRunStatus.FAILED, str(error)[:2000])
            raise

        await self._finish(started, ResearchRunStatus.SUCCEEDED)
        return result

    # ----------------------------------------------------------------------

    async def _finish(
        self,
        started: StartResearchRunOutput,
        status: ResearchRunStatus,
        error_message: str | None = None,
    ) -> None:
        await workflow.execute_activity(
            Activity.FINISH_RESEARCH_RUN,
            FinishResearchRunInput(
                research_run_id=started.research_run_id,
                status=status,
                error_message=error_message,
            ),
            result_type=FinishResearchRunOutput,
            **WRITE_ACTIVITY,
        )

    async def _record(
        self,
        request: ResearchInput,
        started: StartResearchRunOutput,
        records: tuple[ScrapedRecord, ...],
    ) -> ResearchOutput:
        """Archive each record's citation, then write its claims.

        Domain-free: `record.source_url` is guaranteed by ScrapedRecord and the
        ingest activity dispatches on the record's own type, so this loop never
        learns what it is looping over.
        """

        subject_ids: list = []
        skipped: list[str] = []
        new_subjects = 0
        created = corroborated = unchanged = misaligned = 0

        for record in records:
            try:
                archived: ArchiveUrlOutput = await workflow.execute_activity(
                    Activity.ARCHIVE_URL,
                    ArchiveUrlInput(
                        url=record.source_url,
                        kind=SourceKind.WEB_PAGE,
                        title=record.source_title,
                    ),
                    result_type=ArchiveUrlOutput,
                    **FETCH_ACTIVITY,
                )
            except ActivityError:
                # An unfetchable citation is not a workflow failure; it means
                # this one record cannot be evidenced, so it does not get stored.
                workflow.logger.warning("could not archive %s", record.source_url)
                skipped.append(record.source_url)
                continue

            ingested: IngestRecordOutput = await workflow.execute_activity(
                Activity.INGEST_RECORD,
                IngestRecordInput(
                    record=record,
                    source_snapshot_id=archived.source_snapshot_id,
                    research_run_id=started.research_run_id,
                    # self.agent, not a module global: a subclass that swaps the
                    # agent must not attribute its claims to the one it replaced.
                    asserted_by=request.asserted_by or self.agent.name,
                ),
                result_type=IngestRecordOutput,
                **WRITE_ACTIVITY,
            )
            subject_ids.append(ingested.subject_entity_id)
            new_subjects += 1 if ingested.subject_created else 0
            created += ingested.claims_created
            corroborated += ingested.claims_corroborated
            unchanged += ingested.claims_unchanged
            misaligned += ingested.misaligned_count

        return ResearchOutput(
            research_run_id=started.research_run_id,
            records_found=len(records),
            records_new=new_subjects,
            records_already_known=len(subject_ids) - new_subjects,
            claims_created=created,
            claims_corroborated=corroborated,
            claims_unchanged=unchanged,
            misaligned_count=misaligned,
            subject_entity_ids=tuple(subject_ids),
            skipped_urls=tuple(skipped),
        )
