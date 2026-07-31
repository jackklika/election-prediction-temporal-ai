"""The race-structure agent.

An agent rather than an importer because no feed publishes this. The FEC says
who filed; nothing says "Michigan elects a governor in 2026, the primary is in
August and its winner goes to the November general". That lives in encyclopedic
prose, which is what an agent is for.

It reports the components of a race — division, office, cycle, stage — and never
a contest key. Deriving the key is arithmetic, and arithmetic belongs in
`research/structure.py`, where a wrong answer is a bug rather than a
hallucination that mints an unreachable contest.
"""

from __future__ import annotations

from temporalio import workflow

from predictelection.research.structure import ScrapedRaceStructure


with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent

    from predictelection.agents.base import build_research_agent
    from predictelection.clients.anthropic import AnthropicConfig


class StructureFindings(BaseModel):
    """Every contest the agent could describe in one pass."""

    contests: tuple[ScrapedRaceStructure, ...] = Field(
        default=(),
        description=(
            "Every race you can cite. Report the primary and the general as "
            "separate entries when both exist."
        ),
    )


INSTRUCTIONS = """\
You describe the structure of elections — which seats are contested, where, in
what year, and in what round — so races can be compared to one another.

Structure-specific rules:
- A primary and a general are SEPARATE entries. They have different candidates,
  different polls and different outcomes. Reporting them as one loses all three.
  Each party's primary is its own entry too.
- `party` is only for a primary or a caucus. A general election is contested
  between parties, so leave it null there.
- `division_id` is an Open Civic Data ID. For a US House race it is the
  district, 'ocd-division/country:us/state:mi/cd:11', not the state. For a
  statewide race it is the state. For president it is
  'ocd-division/country:us'.
- `office` is the seat alone: 'Governor', not 'Governor of Michigan' and not
  'Michigan Governor 2026'. The place and the year are separate fields.
- Set `advances_to` to 'general' on a primary whose winner goes to a general.
  Leave it null for the final round.
"""


def build_agent(
    config: AnthropicConfig | None = None, *, model=None
) -> Agent[None, StructureFindings]:
    return build_research_agent(
        name="find_race_structure",
        instructions=INSTRUCTIONS,
        output_type=StructureFindings,
        config=config,
        model=model,
    )


structure_agent = build_agent()
"""Module-level because PydanticAIWorkflow.__pydantic_ai_agents__ is read when
the workflow class is defined, so the instance has to exist by import time."""
