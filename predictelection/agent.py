from pydantic import BaseModel, Field
from pydantic_ai import AgentRunResult
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.durable_exec.temporal import LogfirePlugin
from temporalio import workflow


with workflow.unsafe.imports_passed_through():
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from datetime import timedelta, datetime
    import logfire
    from pydantic_ai_harness import CodeMode
    from pydantic_ai.capabilities import WebSearch

    from predictelection.clients.anthropic import AnthropicConfig
    import uuid
    from temporalio.client import Client
    from temporalio.worker import Worker
    from pydantic_ai import Agent
    from pydantic_ai.durable_exec.temporal import (
        PydanticAIPlugin,
        PydanticAIWorkflow,
        TemporalDurability,
    )


_anthropic_config = AnthropicConfig()

model = AnthropicModel(
    _anthropic_config.default_model,
    provider=AnthropicProvider(api_key=_anthropic_config.api_key),
)
logfire.configure()
logfire.instrument_pydantic_ai()


class Debate(BaseModel):
    title: str = Field(description=("Formal name or title of the debate"))
    date: datetime = Field(description=("Approximate time the debate began"))
    youtube_url: str | None = Field(
        default=None,
        description="Link to the video of the full debate, ideally official video from event host",
    )


class DebateResponse(BaseModel):
    debates: list[Debate]


Agent.instrument_all()  # todo put this in a better place
agent = Agent(
    model,
    instructions="You are able to find lists of debates for politicians, try to find all significant or notable debates they have participated in.",
    name="default",
    output_type=DebateResponse,
    capabilities=[
        TemporalDurability(
            activity_config=workflow.ActivityConfig(
                start_to_close_timeout=timedelta(minutes=5)
            )
        ),
        WebSearch(),
        Instrumentation(),
        CodeMode(),
    ],
)


@workflow.defn
class FindDebatesWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, prompt: str) -> DebateResponse:
        result: AgentRunResult[DebateResponse] = await agent.run(prompt)
        return result.output


async def main():
    client = await Client.connect(
        "localhost:7233",
        plugins=[PydanticAIPlugin(), LogfirePlugin()],
    )
    async with Worker(
        client,
        task_queue="default",
        workflows=[FindDebatesWorkflow],
    ):
        output = await client.execute_workflow(
            FindDebatesWorkflow.run,
            args=["abdul el sayed"],
            id=f"debate-{uuid.uuid4()}",
            task_queue="default",
        )
        print(output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
