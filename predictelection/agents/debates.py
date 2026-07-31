"""The debate-finding agent.

Its output type is ScrapedDebate itself rather than a private model, so there is
no mapping step between what the model emits and what gets stored — one schema,
one place for the field descriptions, nothing to drift.

Only what is specific to debates lives here. The capabilities, and the rules
about citing and precision and identifiers that every domain shares, come from
`agents/base.py`.
"""

from __future__ import annotations

from temporalio import workflow

from predictelection.research.debates import ScrapedDebate


with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent

    from predictelection.agents.base import build_research_agent
    from predictelection.clients.anthropic import AnthropicConfig, Effort


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

Debate-specific rules:
- List moderators and panelists under `moderators`, never under `participants`.
  Participants are the people competing against each other.
- A primary debate and a general-election debate are different events even when
  the same people appear in both.
"""


EFFORT: Effort = "high"
"""Above the project default, on the theory that here effort buys recall.

**Not yet established.** The observation behind it: on one subject with an empty
graph, Sonnet 4.6 at `high` reported six debates including three from the
subject's 2018 gubernatorial race, while Sonnet 5 at `medium` reported only the
three 2026 Senate primary debates. That is two variables moving at once — the
model changed with the effort — so it is equally consistent with Sonnet 5
scoping more tightly than 4.6. A controlled run (same model, effort swept) has
not been done.

Set high anyway because the asymmetry favours it: finding a debate is this
agent's entire job, a debate nobody reports is indistinguishable from one that
never happened, and at `high` a run still finishes in a few minutes. Lower it
once there is a measurement, not before.

Effort is usually a cost knob. On an agent whose output is coverage it is
plausibly a recall knob, which is why it is per domain: the structure agent
answers a far more scoped question.

ANTHROPIC_EFFORT still overrides this, so a sweep measures this agent too.
"""


def build_agent(
    config: AnthropicConfig | None = None, *, model=None
) -> Agent[None, DebateFindings]:
    """This domain's agent, so a test can stub the model without restating it.

    Injected at construction rather than swapped with `agent.override()`:
    TemporalDurability turns model calls into activities, and the worker's
    activity task does not inherit the context variable override sets, so the
    real provider gets used anyway — silently, over the network, at cost.
    """

    return build_research_agent(
        name="find_debates",
        instructions=INSTRUCTIONS,
        output_type=DebateFindings,
        config=config,
        model=model,
        effort=EFFORT,
    )


debate_agent = build_agent()
"""Module-level because PydanticAIWorkflow.__pydantic_ai_agents__ is read when
the workflow class is defined, so the instance has to exist by import time.

Importing this module therefore needs ANTHROPIC_API_KEY. Nothing that only
handles contracts or database writes should import it — that is why the
activities live in a separate package.
"""
