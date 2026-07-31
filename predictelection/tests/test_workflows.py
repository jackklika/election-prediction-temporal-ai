"""The workflow end to end, with the LLM stubbed and everything else real.

Runs on Temporal's time-skipping test server, so it needs no compose Temporal —
`make test-db` still only requires Postgres and MinIO. The database and object
store are real, because the point is to prove the orchestration actually works.

This is the only test covering three things nothing else can: that the contract
models survive Temporal's data converter, that dispatching activities by string
name resolves to the registered implementations, and that a debate whose citation
cannot be fetched is skipped rather than failing the run or — worse — being stored
with a source nobody can check.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import uuid

import httpx
import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from temporalio import workflow

from predictelection.activities.contracts import (
    ResearchDebatesInput,
    ResearchDebatesOutput,
)
from predictelection.agents.debates import build_agent
from predictelection.workflows.debates import ResearchDebatesWorkflow
from predictelection.activities.research import ResearchActivities
from predictelection.sql import (
    Claim,
    ClaimAssertion,
    Entity,
    ResearchRun,
    create_schema,
    ontology_alignment_score,
)
from predictelection.tests.conftest import _drop_suite_tables


pytestmark = pytest.mark.postgres


GOOD_URL = "https://example.test/mi-debate"
BAD_URL = "https://example.test/gone"

PAGE = b"<html><body><h1>Michigan governor debate</h1></body></html>"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _http() -> httpx.Client:
    """Serves one citation and 404s the other."""

    def handle(request: httpx.Request) -> httpx.Response:
        if str(request.url) == BAD_URL:
            return httpx.Response(404)
        return httpx.Response(200, content=PAGE, headers={"content-type": "text/html"})

    return httpx.Client(transport=httpx.MockTransport(handle))


def _findings():
    """Two debates: one citable, one not."""

    from predictelection.agents.debates import DebateFindings
    from predictelection.research.debates import ScrapedDebate
    from predictelection.research.scraped import ScrapedEntity

    return DebateFindings(
        debates=(
            ScrapedDebate(
                title="2026 Michigan Gubernatorial Debate",
                source_url=GOOD_URL,
                starts_at=datetime(2026, 9, 15, 21, 0, tzinfo=UTC),
                participants=(ScrapedEntity(name="Abdul El-Sayed"),),
                jurisdiction=ScrapedEntity(name="Michigan"),
            ),
            ScrapedDebate(
                title="A debate whose source has since vanished",
                source_url=BAD_URL,
                starts_at=datetime(2026, 8, 1, tzinfo=UTC),
                participants=(ScrapedEntity(name="Abdul El-Sayed"),),
            ),
        )
    )


_CALLS: list[int] = []
_PROMPTS: list[str] = []


def _stub_model():
    """A model that answers with fixed findings instead of calling Anthropic.

    Injected at agent construction, not via agent.override(): the model call runs
    inside a Temporal activity whose task does not inherit the overriding context
    variable, so an override silently falls back to the real provider and bills
    for it. _CALLS records invocations so the test can prove the stub was used.
    """

    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    def respond(messages, info: AgentInfo) -> ModelResponse:
        _CALLS.append(1)
        # Record the prompt so the test can prove the workflow fetched existing
        # events and handed them to the agent.
        _PROMPTS.append(
            "\n".join(
                str(getattr(part, "content", ""))
                for message in messages
                for part in getattr(message, "parts", [])
            )
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    _findings().model_dump(mode="json"),
                )
            ]
        )

    return FunctionModel(respond)


STUB_AGENT = build_agent(model=_stub_model())
"""Built with an injected model, so this never constructs an Anthropic client."""


@workflow.defn(name="StubbedResearchDebates", sandboxed=False)
class StubbedResearchDebatesWorkflow(ResearchDebatesWorkflow):
    """The real workflow with only its agent swapped.

    Module scope because Temporal rejects local classes, and a subclass rather
    than a rewrite so the orchestration under test is the production code.
    """

    agent = STUB_AGENT
    __pydantic_ai_agents__ = [STUB_AGENT]

    @workflow.run
    async def run(self, request: ResearchDebatesInput) -> ResearchDebatesOutput:
        return await super().run(request)


@pytest.fixture
async def workflow_env(pytestconfig, anyio_backend):
    """Temporal's time-skipping test server.

    Downloads a test-server binary the first time it runs, so an offline machine
    gets the same skip treatment as a missing database rather than an error.
    """

    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

    try:
        env = await WorkflowEnvironment.start_time_skipping(
            plugins=[PydanticAIPlugin()]
        )
    except Exception as error:  # noqa: BLE001 - startup failure mode is broad
        message = (
            f"Temporal test server unavailable: {error.__class__.__name__}: {error}"
        )
        if pytestconfig.getoption("--require-postgres"):
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    try:
        yield env
    finally:
        await env.shutdown()


@pytest.fixture
def committed_sessions(postgres_engine: Engine):
    """Sessions on the engine itself, and a clean database afterwards.

    Unlike every other test here, this one cannot borrow the rolled-back `session`
    fixture: Temporal runs the sync activities on worker threads, and a SQLAlchemy
    Connection is not safe to share across threads. Lending one out produced a
    seven-minute run and results that did not match what the activities wrote.

    So activities get the engine's own pooled connections and really commit, and
    the schema is rebuilt afterwards so later tests still see a pristine, seeded
    database rather than this test's leftovers.
    """

    yield sessionmaker(bind=postgres_engine, expire_on_commit=False)

    _drop_suite_tables(postgres_engine)
    create_schema(postgres_engine)


@pytest.fixture
def read_session(postgres_engine: Engine):
    """A fresh session for assertions, so it sees what the activities committed."""

    with Session(postgres_engine, expire_on_commit=False) as reader:
        yield reader


async def _run_workflow(
    client: Client,
    activities: ResearchActivities,
    request: ResearchDebatesInput,
):
    task_queue = f"test-{uuid.uuid4()}"
    executor = ThreadPoolExecutor(max_workers=4)
    try:
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[StubbedResearchDebatesWorkflow],
            activities=activities.all(),
            activity_executor=executor,
        ):
            return await client.execute_workflow(
                StubbedResearchDebatesWorkflow.run,
                request,
                id=f"research-{uuid.uuid4()}",
                task_queue=task_queue,
            )
    finally:
        executor.shutdown(wait=False)


@pytest.mark.anyio
async def test_the_workflow_stores_only_what_it_can_cite(
    workflow_env, committed_sessions, object_store, read_session: Session
) -> None:
    activities = ResearchActivities(
        session_factory=committed_sessions, store=object_store, http=_http()
    )

    result = await _run_workflow(
        workflow_env.client,
        activities,
        ResearchDebatesInput(subject="Abdul El-Sayed"),
    )

    assert _CALLS, "the stub model was never called, so the real provider ran"
    # nothing was recorded before this run, so the context block is absent
    assert not any("ALREADY RECORDED" in prompt for prompt in _PROMPTS)
    assert result.debates_found == 2
    assert result.debates_new == 1
    assert result.debates_already_known == 0
    assert result.skipped_urls == (BAD_URL,)
    assert result.claims_created > 0
    assert result.claims_corroborated == 0
    assert result.misaligned_count == 0

    # the vanished source left nothing behind
    titles = set(read_session.scalars(select(Entity.canonical_name)))
    assert "A debate whose source has since vanished" not in titles

    # and the run itself is closed out, attributed, and tied to this execution
    run = read_session.get(ResearchRun, result.research_run_id)
    assert run is not None
    assert run.task_type == "find_debates"
    assert run.input_data == {"subject": "Abdul El-Sayed"}
    assert run.workflow_id is not None
    assert run.workflow_run_id is not None
    assert (
        read_session.scalar(
            select(func.count(ClaimAssertion.id)).where(
                ClaimAssertion.research_run_id == run.id
            )
        )
        == result.claims_recorded
    )


@pytest.mark.anyio
async def test_rerunning_the_workflow_does_not_duplicate(
    workflow_env, committed_sessions, object_store, read_session: Session
) -> None:
    """Researching the same subject twice adds evidence, not duplicate facts.

    Both fixes are visible here. The second execution gets its *own* ResearchRun,
    so "what did this run find" stays answerable and a per-run alignment score
    means something — keying the run on {task_type, subject} instead collapsed
    every run for a subject into one row forever. And it reports the debate as
    already known rather than counting it as ingested, which is the difference
    between "found 5 debates" and "learned nothing".
    """

    activities = ResearchActivities(
        session_factory=committed_sessions, store=object_store, http=_http()
    )
    request = ResearchDebatesInput(subject="Abdul El-Sayed")

    first = await _run_workflow(workflow_env.client, activities, request)
    claims_after_first = read_session.scalar(select(func.count(Claim.id)))
    entities_after_first = read_session.scalar(select(func.count(Entity.id)))

    second = await _run_workflow(workflow_env.client, activities, request)

    # its own run, so the two are still tellable apart
    assert second.research_run_id != first.research_run_id

    # but the same entities, and no new propositions
    assert second.event_ids == first.event_ids
    assert read_session.scalar(select(func.count(Claim.id))) == claims_after_first
    assert read_session.scalar(select(func.count(Entity.id))) == entities_after_first

    # and it says so plainly instead of claiming to have ingested a debate
    assert second.debates_new == 0
    assert second.debates_already_known == 1
    assert second.claims_created == 0
    assert second.claims_corroborated == first.claims_created

    # per-run alignment is answerable again, rather than averaged across history
    for run_id in (first.research_run_id, second.research_run_id):
        assert ontology_alignment_score(read_session, research_run_id=run_id) == 1.0


@pytest.mark.anyio
async def test_the_second_run_is_shown_the_first_runs_titles(
    workflow_env, committed_sessions, object_store, read_session: Session
) -> None:
    """The fix for 11 event entities standing for 6 real debates.

    The agent re-described each debate because it could not see what it had
    already written. The workflow now looks that up and puts it in the prompt, so
    the model can echo a title back verbatim instead of inventing a new phrasing.

    Fetched by the workflow rather than by an agent tool: TemporalDurability runs
    each agent step as an activity, and an activity cannot call execute_activity,
    so a tool for this hangs.
    """

    activities = ResearchActivities(
        session_factory=committed_sessions, store=object_store, http=_http()
    )
    request = ResearchDebatesInput(subject="Abdul El-Sayed")

    _PROMPTS.clear()
    await _run_workflow(workflow_env.client, activities, request)
    first_prompts = list(_PROMPTS)

    _PROMPTS.clear()
    await _run_workflow(workflow_env.client, activities, request)
    second_prompts = list(_PROMPTS)

    assert not any("ALREADY RECORDED" in p for p in first_prompts)
    context = "\n".join(second_prompts)
    assert "ALREADY RECORDED" in context
    assert "2026 Michigan Gubernatorial Debate" in context
