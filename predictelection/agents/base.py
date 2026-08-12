"""How every research agent is built, and the rules all of them follow.

Two things live here because copying them is how they drift:

- **The capability block.** TemporalDurability, web search, instrumentation and
  CodeMode are not per-domain choices; they are what "a research agent in this
  project" means. Pasted into each domain, one of them eventually gets left out
  of one agent and nobody notices which.
- **SHARED_INSTRUCTIONS.** Cite everything, do not over-claim precision, do not
  guess identifiers, prefer primary sources. These are properties of
  `ScrapedRecord` and `ScrapedEntity`, not of debates — the same argument
  `research/scraped.py` makes for putting `source_url` on the base class. A
  domain that restated them could quietly restate one of them wrongly.

The agent only reads and reports. It never touches the database: writing is an
activity, so a retried or replayed workflow cannot re-run the LLM but can safely
re-run the write.
"""

from __future__ import annotations

from typing import TypeVar, cast

from temporalio import workflow
from temporalio.common import RetryPolicy


with workflow.unsafe.imports_passed_through():
    from datetime import timedelta

    from pydantic import BaseModel
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import Instrumentation, WebSearch
    from pydantic_ai.durable_exec.temporal import TemporalDurability
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai_harness import CodeMode

    from predictelection.clients.anthropic import AnthropicConfig, Effort


OutputT = TypeVar("OutputT", bound=BaseModel)


SHARED_INSTRUCTIONS = """\
Rules that matter more than completeness, and apply to everything you report:
- Cite every record with the source_url you actually read it on. Anything you
  cannot cite must be left out rather than reported.
- Never claim more precision than the source gave. If it states only a date, say
  'day'. Use the article's publication date to resolve relative wording like
  "last night".
- Only give an identifier — wikidata_id, ocd_id, fec_id — when you are certain
  it is the right one. A wrong identifier merges two different things, which is
  far more damaging than leaving it null.
- Prefer official or primary sources over aggregators.
- REPORT EVERYTHING YOU FIND, including anything already listed under ALREADY
  RECORDED. Re-reporting something already in the graph is not a duplicate and
  is not wasted work: it is how a second source corroborates a fact, and it is
  the only way to tell "there is nothing more to find" apart from "you did not
  look". Never leave a record out because it is already listed.
- When you report something that IS on that list, use its name from the list,
  character for character. Reporting it under a new wording is what creates a
  duplicate nobody can automatically merge.
"""

DEFAULT_EFFORT: Effort = "medium"
"""How hard the model works per request, when nothing overrides it.

The API default is `high`, and leaving it there is the single largest avoidable
cost in this project — larger than the choice of model. Effort governs thinking
depth *and* how much the model explores before answering, so on a research agent
it multiplies searches and page reads, not just tokens.

`medium` because this is extraction, not open-ended reasoning: the agents read
sources and fill in a schema whose fields are already specified. The judgement
that matters — whether an identifier is certain enough to state, whether the
source gave a date or a time — is a matter of following the instructions in
SHARED_INSTRUCTIONS rather than of reasoning depth.

Raise it per agent for a genuinely harder domain rather than globally, and note
that `xhigh` is rejected outright by models that do not support it.
"""

AGENT_TIMEOUT = timedelta(minutes=20)
"""How long one model request may take before Temporal calls it dead.

Generous because a web-searching agent legitimately takes minutes: it issues
searches, reads pages, and only then produces output. Five minutes looked ample
and was not — a second run on an already-researched subject, whose prompt
carries the ALREADY RECORDED block, ran past it and every attempt was killed
mid-flight.

The failure is expensive and quiet in the wrong way. The activity heartbeats
throughout, so it is visibly alive right up until the timeout fires; nothing
distinguishes "still working" from "about to be killed and retried", and each
retry pays for the whole request again.
"""

AGENT_HEARTBEAT_TIMEOUT = timedelta(seconds=60)
"""A genuinely wedged request should still be caught quickly."""

AGENT_MAX_ATTEMPTS = 3
"""Unlimited retries on a model call is a way to spend money in a loop.

Temporal's default is unbounded, so a request that always exceeds the timeout —
because the timeout is wrong, not because the call is flaky — retries forever,
paying the provider each time and never surfacing an error. Three attempts is
enough for a rate limit or a 529, and it stops.
"""


def build_research_agent(
    *,
    name: str,
    instructions: str,
    output_type: type[OutputT],
    config: AnthropicConfig | None = None,
    model=None,
    effort: Effort | None = None,
) -> Agent[None, OutputT]:
    """Construct a research agent with the shared rules and capabilities.

    `model` exists so tests can supply a stub. It has to be injected here rather
    than swapped later with `agent.override()`: TemporalDurability turns model
    calls into activities, and the worker's activity task does not inherit the
    context variable that override sets, so the real provider gets used anyway —
    silently, over the network, at cost.

    `effort` resolves ANTHROPIC_EFFORT, then this argument, then
    `DEFAULT_EFFORT`. The environment wins deliberately: a domain that pins its
    own level should still be swept along with everything else, or a sweep
    silently measures only the agents that happened not to pin one. The config
    step applies only when a real model is being built, so a stubbed agent still
    needs no API key.

    No tools are registered, and none should be. TemporalDurability runs each
    agent step *as an activity*, and an activity cannot start another, so a tool
    that looks something up hangs rather than erroring. Workflows fetch context
    and pass it into the prompt instead. CodeMode would also hide any registered
    tool behind a sandboxed run_code, where the model never sees it.
    """

    if model is None:
        # Only resolved when a real model is wanted, so a stubbed agent needs no
        # API key at all.
        settings = config or AnthropicConfig()  # ty: ignore[missing-argument]
        effort = settings.effort or effort
        model = AnthropicModel(
            settings.default_model,
            provider=AnthropicProvider(api_key=settings.api_key),
        )
    # cast because the checker cannot follow output_type= through Agent's
    # overloads; output_type is what it actually returns.
    agent = Agent(
        model,
        name=name,
        instructions=f"{instructions.rstrip()}\n\n{SHARED_INSTRUCTIONS}",
        output_type=output_type,
        # Every anthropic_* setting is namespaced precisely so it can ride along
        # with a non-Anthropic model, so a stubbed FunctionModel ignores this
        # rather than choking on it.
        model_settings=AnthropicModelSettings(
            anthropic_effort=effort or DEFAULT_EFFORT
        ),
        capabilities=[
            TemporalDurability(
                activity_config=workflow.ActivityConfig(
                    start_to_close_timeout=AGENT_TIMEOUT,
                    heartbeat_timeout=AGENT_HEARTBEAT_TIMEOUT,
                    retry_policy=RetryPolicy(maximum_attempts=AGENT_MAX_ATTEMPTS),
                )
            ),
            WebSearch(),
            Instrumentation(),
            CodeMode(),
        ],
    )
    return cast("Agent[None, OutputT]", agent)
