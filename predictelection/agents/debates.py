"""The debate-finding agent.

Its output type is ScrapedDebate itself rather than a private model, so there is
no mapping step between what the model emits and what gets stored — one schema,
one place for the field descriptions, nothing to drift.

The agent only reads and reports. It never touches the database: writing is an
activity, so a retried or replayed workflow cannot re-run the LLM but can safely
re-run the write.
"""

from __future__ import annotations

from typing import cast

from temporalio import workflow

from predictelection.research.debates import ScrapedDebate


with workflow.unsafe.imports_passed_through():
    from datetime import timedelta

    from pydantic import BaseModel, Field
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import Instrumentation, WebSearch
    from pydantic_ai.durable_exec.temporal import TemporalDurability
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai_harness import CodeMode

    from predictelection.clients.anthropic import AnthropicConfig


class DebateFindings(BaseModel):
    """Everything the agent found in one pass."""

    debates: tuple[ScrapedDebate, ...] = Field(
        default=(),
        description=(
            "Every notable debate you can cite. Omit any you cannot attribute to "
            "a source URL rather than guessing one."
        ),
    )


INSTRUCTIONS = """\
You find debates that a politician took part in, and report them so they can be
stored as verifiable facts.

Rules that matter more than completeness:
- Cite every debate with the source_url you actually read it on. A debate you
  cannot cite must be left out.
- Never claim more precision than the source gave. If it says only a date, set
  starts_at_precision to 'day'.
- Only give a wikidata_id when you are certain it identifies that exact person.
- Prefer official or primary sources over aggregators.
"""


def build_agent(
    config: AnthropicConfig | None = None, *, model=None
) -> Agent[None, DebateFindings]:
    """Construct the debate agent.

    `model` exists so tests can supply a stub. It has to be injected here rather
    than swapped later with `agent.override()`: TemporalDurability turns model
    calls into activities, and the worker's activity task does not inherit the
    context variable that override sets, so the real provider gets used anyway —
    silently, over the network, at cost.
    """

    if model is None:
        # Only resolved when a real model is wanted, so a stubbed agent needs no
        # API key at all.
        settings = config or AnthropicConfig()  # ty: ignore[missing-argument]
        model = AnthropicModel(
            settings.default_model,
            provider=AnthropicProvider(api_key=settings.api_key),
        )
    # cast because the checker cannot follow output_type= through Agent's
    # overloads; DebateFindings is what it actually returns.
    agent = Agent(
        model,
        name="find_debates",
        instructions=INSTRUCTIONS,
        output_type=DebateFindings,
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
    return cast("Agent[None, DebateFindings]", agent)


debate_agent = build_agent()
"""Module-level because PydanticAIWorkflow.__pydantic_ai_agents__ is read when
the workflow class is defined, so the instance has to exist by import time.

Importing this module therefore needs ANTHROPIC_API_KEY. Nothing that only
handles contracts or database writes should import it — that is why the
activities live in a separate package.
"""
