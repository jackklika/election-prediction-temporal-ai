"""The campaign-finance agent.

An agent rather than an importer, *for now*. FEC bulk data is the authoritative
source for itemized contributions and it is deterministic, structured and huge —
it belongs in `importers/` when someone wants the whole picture. What an agent
adds is the reporting around the filings: who a super PAC is aligned with, that
a seven-figure buy ran against a candidate rather than for them, and money in
races the FEC does not cover at all — state offices and ballot measures.

The two rules in the instructions are the two ways this domain goes wrong:
conflating an independent expenditure with a contribution, and assuming money
mentioning a candidate was spent supporting them.
"""

from __future__ import annotations

from temporalio import workflow

from predictelection.research.donations import ScrapedDonation


with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent

    from predictelection.agents.base import build_research_agent
    from predictelection.clients.anthropic import AnthropicConfig


class DonationFindings(BaseModel):
    """One pass over a race's money."""

    donations: tuple[ScrapedDonation, ...] = Field(
        default=(),
        description=(
            "Every contribution, independent expenditure, in-kind gift or loan "
            "the source states. One entry per reported transaction."
        ),
    )


INSTRUCTIONS = """\
You report money given in aid of, or against, candidates in a race.

Two distinctions carry most of the value, and getting them wrong is worse than
reporting nothing:

- A CONTRIBUTION goes to the candidate or their committee and is capped. An
  INDEPENDENT EXPENDITURE is spent about them by someone who may not coordinate
  with them, and is uncapped. They are legally different things. Use `kind` to
  say which; do not describe a super PAC's ad buy as a contribution.
- Money that mentions a candidate is not necessarily FOR them. Set `supporting`
  to false when an expenditure ran against the recipient, true when it ran for
  them, and leave it null for direct contributions and whenever the source does
  not say. Assuming support inverts the fact in the most misleading direction.

The rest:

- Report the amount exactly as published. Never estimate, never convert a
  described sum ("maxed out", "six figures") into a number — leave `amount`
  null and let the absence be honest.
- The donor is whoever the filing names. A PAC is the donor; the people who
  funded the PAC are separate donations to the PAC, and only report those if the
  source states them.
- NEVER promote a vague date to a precise one. "in July" is 2026-07-01 with
  precision 'month'. A cycle total with no date is precision 'year', or a null
  date if even the cycle is unclear.
- A refund or a returned contribution is its own record, not an edit to the
  original. Report it with the amount as published.
- Totals are not transactions. "raised $4M this quarter" is a summary, not a
  donation, and has no donor — do not invent one to make it fit.
"""


def build_agent(
    config: AnthropicConfig | None = None, *, model=None
) -> Agent[None, DonationFindings]:
    return build_research_agent(
        name="find_donations",
        instructions=INSTRUCTIONS,
        output_type=DonationFindings,
        config=config,
        model=model,
        # The project default. Unlike candidacies, this is transcription rather
        # than reading comprehension — the facts are stated plainly in filings
        # and coverage, so the budget is better spent on coverage than on
        # reasoning about them.
    )


donation_agent = build_agent()
"""Module-level because PydanticAIWorkflow.__pydantic_ai_agents__ is read when
the workflow class is defined, so the instance has to exist by import time."""
