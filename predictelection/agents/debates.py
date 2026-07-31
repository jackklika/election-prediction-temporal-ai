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
    from datetime import datetime, timedelta

    from pydantic import BaseModel, Field
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import Instrumentation, WebSearch
    from pydantic_ai.durable_exec.temporal import TemporalDurability
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai_harness import CodeMode

    from predictelection.clients.anthropic import AnthropicConfig
    from predictelection.sql import EntityKind


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

Before reporting a debate, check what is already recorded. In the code sandbox:

    await known_events(occurred_after="2026-09-01", occurred_before="2026-09-30")

If a debate already exists for that date, REPORT IT UNDER THE EXISTING TITLE,
character for character. Re-describing a debate that is already recorded creates
a duplicate that nobody can automatically merge — the same event under two names
is worse than not reporting it twice. Search by date rather than by title: two
debates a month apart share nearly all their words, so the date is what tells
them apart.

Rules that matter more than completeness:
- Cite every debate with the source_url you actually read it on. A debate you
  cannot cite must be left out.
- Never claim more precision than the source gave. If it says only a date, set
  starts_at_precision to 'day'. Use the article's publication date to resolve
  relative wording like "last night".
- List moderators and panelists under `moderators`, never under `participants`.
  Participants are the people competing against each other.
- Only give an identifier — wikidata_id, ocd_id, fec_id — when you are certain
  it is the right one. A wrong identifier merges two different people, which is
  far more damaging than leaving it null.
- Prefer official or primary sources over aggregators.
"""


async def known_events(
    name: str | None = None,
    occurred_after: datetime | None = None,
    occurred_before: datetime | None = None,
) -> list[dict[str, str]]:
    """Look up debates already recorded, so you can reuse their exact titles.

    Call this before reporting a debate. Search by the date it happened —
    two debates a month apart share nearly all their words, while the same
    debate described twice shares a date, so the date discriminates where the
    title cannot.

    Returns the stored title for each match. If one is the debate you are about
    to report, use that title verbatim instead of writing your own.
    """

    from temporalio import workflow as _workflow

    from predictelection.activities.contracts import (
        FindEntitiesInput,
        FindEntitiesOutput,
    )

    found: FindEntitiesOutput = await _workflow.execute_activity(
        "find_entities",
        FindEntitiesInput(
            name=name,
            kind=EntityKind.EVENT,
            occurred_after=occurred_after,
            occurred_before=occurred_before,
        ),
        result_type=FindEntitiesOutput,
        start_to_close_timeout=timedelta(seconds=30),
    )
    return [
        {
            "title": match.canonical_name,
            "occurred_at": (
                match.occurred_at.date().isoformat() if match.occurred_at else "unknown"
            ),
        }
        for match in found.matches
    ]


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
    # overloads; DebateFindings is what it actually returns. The ignore is the
    # same limitation for tools= — verified registered at runtime.
    agent = Agent(  # ty: ignore[no-matching-overload]
        model,
        name="find_debates",
        instructions=INSTRUCTIONS,
        output_type=DebateFindings,
        tools=[known_events],
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
