"""`ResearchWorkflow.ask`, without Temporal or a database.

The same guard is covered end to end in `test_workflows.py`, but that test costs
a test server and a minute of wall clock. This one runs in `make test`, which
matters because the code it covers is the code that must not be wrong: `ask`
runs inside workflow code, where anything other than an `ApplicationError`
becomes a workflow-task failure and is retried forever rather than reported.

No API key needed: the agent is built with a stub model.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from temporalio.exceptions import ApplicationError

from predictelection.agents.base import build_research_agent
from predictelection.workflows.base import UNFINISHED_REASONS, ResearchWorkflow


class Findings(BaseModel):
    answer: str = ""


def _agent(finish_reason: str | None, state: str = "complete"):
    def respond(messages: Any, info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart('{"answer": "ok"}')],
            finish_reason=finish_reason,  # ty: ignore[invalid-argument-type]
            state=state,  # ty: ignore[invalid-argument-type]
        )

    return build_research_agent(
        name="guarded_agent",
        instructions="Domain rules.",
        output_type=Findings,
        model=FunctionModel(respond),
    )


async def _ask(finish_reason: str | None, state: str = "complete") -> Any:
    """`ask` on a workflow that has nothing but an agent.

    The agent goes on the class, not the instance, because that is where
    `ResearchWorkflow` declares it and how every real workflow sets it — read
    through the class so a stub cannot end up attributing its claims to the
    agent it replaced. Not registered with Temporal: `ask` only awaits the
    agent, so none of the workflow machinery is needed to exercise it.
    """

    guarded = type(
        "_Guarded",
        (ResearchWorkflow,),
        {"task_type": "guard_test", "agent": _agent(finish_reason, state)},
    )
    return await guarded().ask("anything")


@pytest.mark.anyio
@pytest.mark.parametrize("finish_reason", ["stop", "tool_call", None])
async def test_a_finished_answer_is_returned(finish_reason: str | None) -> None:
    """The two ways a response ends on purpose, plus a model that says nothing."""

    assert (await _ask(finish_reason)).answer == "ok"


@pytest.mark.anyio
@pytest.mark.parametrize("finish_reason", sorted(UNFINISHED_REASONS))
async def test_an_unfinished_answer_raises_rather_than_returning_it(
    finish_reason: str,
) -> None:
    """`length` is the one that bit, but a filtered or errored response is the
    same shape of lie: a findings object whose fields fell back to defaults."""

    with pytest.raises(ApplicationError) as failure:
        await _ask(finish_reason)
    assert "did not finish its answer" in str(failure.value)
    assert finish_reason in str(failure.value)
    assert failure.value.non_retryable, (
        "a retryable failure here would re-run a model call that already "
        "exhausted its budget, and pay for it each time"
    )


@pytest.mark.anyio
async def test_an_incomplete_state_raises_even_when_the_reason_looks_fine() -> None:
    """pydantic-ai documents `state`, not `finish_reason`, as the field that says
    whether the response is finished. Both are checked, so a provider that fills
    in one and not the other cannot slip an empty answer through."""

    with pytest.raises(ApplicationError, match="did not finish"):
        await _ask("stop", state="incomplete")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
