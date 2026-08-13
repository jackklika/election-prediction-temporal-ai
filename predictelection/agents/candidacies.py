"""The candidacy-and-endorsement agent.

An agent rather than an importer because a candidacy's *timeline* is prose. A
results table gives counts and a roster heading gives the outcome — both
deterministic, both already imported — but "dropped out in mid-July, prompting
Crowley to re-enter the race ten days after withdrawing" is a sentence, and
turning it into intervals is reading comprehension.

It reports no numbers and no keys: vote counts belong to the results importer,
and the contest key is derived from the components it reports. What it uniquely
supplies is *when* each candidacy was live and *who* backed whom over which
period.
"""

from __future__ import annotations

from temporalio import workflow

from predictelection.research.candidacies import ScrapedCandidacy
from predictelection.research.endorsements import ScrapedEndorsement


with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent

    from predictelection.agents.base import build_research_agent
    from predictelection.clients.anthropic import AnthropicConfig


class CandidacyFindings(BaseModel):
    """One pass over a race: who ran when, and who endorsed whom when."""

    candidacies: tuple[ScrapedCandidacy, ...] = Field(
        default=(),
        description=(
            "Every candidate in the race, including those who withdrew and "
            "those who were disqualified. One entry per person per contest."
        ),
    )
    endorsements: tuple[ScrapedEndorsement, ...] = Field(
        default=(),
        description=(
            "Endorsements the source states, including ones later withdrawn or "
            "switched. One entry per endorsement period."
        ),
    )


INSTRUCTIONS = """\
You reconstruct the timeline of a race: who was a candidate over which periods,
and who endorsed whom over which periods.

The whole point is the *timeline*, so these rules matter more than coverage:

- A candidate who withdrew and later re-entered has TWO stints, not one. The
  first ends when they withdrew; the second begins when they re-entered.
  Collapsing them into one period asserts they were running during a time they
  were publicly backing somebody else.
- A withdrawal is never a deletion. Report the stint that ended, with its end
  date, and let it stand. Someone who left the race in June was still a
  candidate in May, and the record has to keep saying so.
- Set `remained_on_ballot` when the source says a withdrawn candidate stayed on
  the ballot. That is why results can show votes for someone who quit.
- NEVER promote a vague date to a precise one. "mid-July 2026" is
  `starts`/`entered_on` of 2026-07-01 with precision 'month', NOT July 15 with
  precision 'day'. If the source only gives a month, say month.
- A relative date — "ten days after withdrawing", "the following week" — may
  only be reported when the source also states the date it is relative to and
  the arithmetic is unambiguous. Otherwise leave the date null. A null date is
  a fact about what the source said; an invented date is a false fact.
- An endorsement that was later switched or withdrawn is TWO records: the
  original over the period it held, and a second with strength 'withdrawn' over
  the later period. Never one record that gets edited.
- `endorser_kind` is 'organization' for newspapers, unions and interest groups,
  'party' for party organs, 'person' for individuals.
- Report vote totals nowhere. They are imported from the results table
  separately and an approximate number from you would compete with an exact one.
"""


def build_agent(
    config: AnthropicConfig | None = None, *, model=None
) -> Agent[None, CandidacyFindings]:
    return build_research_agent(
        name="find_candidacies",
        instructions=INSTRUCTIONS,
        output_type=CandidacyFindings,
        config=config,
        model=model,
        # Coverage of a whole field plus the endorsement graph, and the
        # timeline detail lives in body prose rather than in tables — the same
        # argument that put the debates agent above the default.
        effort="high",
    )


candidacy_agent = build_agent()
"""Module-level because PydanticAIWorkflow.__pydantic_ai_agents__ is read when
the workflow class is defined, so the instance has to exist by import time."""
