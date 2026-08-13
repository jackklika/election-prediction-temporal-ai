"""How research agents are constructed.

Effort is the single largest avoidable cost here — larger than the choice of
model, because it governs how much the agent explores before answering, not just
how many tokens it emits. The API default is `high`, so *not* setting it is a
decision, and a silent one. These tests make the setting explicit enough that
dropping it shows up as a failure rather than as a bill.

No API key needed: every agent below is built with a stub model.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart

from predictelection.agents.base import (
    AGENT_MAX_TOKENS,
    DEFAULT_EFFORT,
    SHARED_INSTRUCTIONS,
    build_research_agent,
)
from predictelection.clients.anthropic import AnthropicConfig


class Findings(BaseModel):
    answer: str = ""


def _stub() -> FunctionModel:
    def respond(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("ok")])

    return FunctionModel(respond)


def _build(**kwargs):
    return build_research_agent(
        name="test_agent",
        instructions="Domain rules.",
        output_type=Findings,
        model=_stub(),
        **kwargs,
    )


def test_effort_is_set_rather_than_inherited_from_the_api_default() -> None:
    """The API default is `high`; leaving it there is the expensive choice."""

    agent = _build()
    assert agent.model_settings is not None
    assert agent.model_settings["anthropic_effort"] == DEFAULT_EFFORT


def test_the_default_is_not_the_api_default() -> None:
    """If DEFAULT_EFFORT ever becomes `high`, this stops being a saving.

    Pinned so raising it is a deliberate edit with a failing test attached,
    rather than a quiet reversion to what the API would have done anyway.
    """

    assert DEFAULT_EFFORT == "medium"


def test_a_domain_can_raise_effort_for_a_harder_task() -> None:
    """Per-agent rather than global: only some domains need more."""

    agent = _build(effort="high")
    assert agent.model_settings is not None
    assert agent.model_settings["anthropic_effort"] == "high"


def test_config_supplies_effort_so_a_sweep_needs_no_code_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ANTHROPIC_EFFORT is readable, so cost/quality can be swept from the env."""

    monkeypatch.setenv("ANTHROPIC_EFFORT", "low")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    assert AnthropicConfig().effort == "low"  # ty: ignore[missing-argument]


def test_effort_is_unset_by_default_so_agents_choose_their_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None means "no opinion" — which is what lets a domain pin its own level."""

    monkeypatch.delenv("ANTHROPIC_EFFORT", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    assert AnthropicConfig().effort is None  # ty: ignore[missing-argument]


def test_a_sweep_overrides_a_domain_that_pinned_its_own_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise a sweep silently measures only the unpinned agents.

    The debates agent pins `high` because effort buys it recall. A sweep that
    could not reach it would report numbers for the structure agent alone while
    appearing to cover both.
    """

    monkeypatch.setenv("ANTHROPIC_EFFORT", "low")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    config = AnthropicConfig()  # ty: ignore[missing-argument]

    # Mirrors build_research_agent's resolution: environment beats the pin.
    pinned = "high"
    assert (config.effort or pinned) == "low"


def test_an_unknown_effort_level_is_refused_at_config_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad level is a 400 on every single request — catch it at startup."""

    monkeypatch.setenv("ANTHROPIC_EFFORT", "extreme")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    with pytest.raises(ValueError):
        AnthropicConfig()  # ty: ignore[missing-argument]


def test_shared_rules_are_prepended_to_every_domain() -> None:
    """A domain inherits the citation and identifier rules instead of restating."""

    agent = _build()
    raw = agent._instructions
    instructions = "\n".join(raw) if isinstance(raw, list) else str(raw or "")
    assert "Domain rules." in instructions
    assert SHARED_INSTRUCTIONS.strip() in instructions


def test_the_debates_agent_runs_above_the_project_default() -> None:
    """Pinned deliberately: for this agent effort is plausibly recall, not cost.

    The supporting observation is confounded — the model changed along with the
    effort — so this is a cautious default rather than a measured one. Pinned so
    that lowering it is a deliberate edit rather than drift, and so the reasoning
    in `debates.EFFORT` gets read first.
    """

    from predictelection.agents import debates

    assert debates.EFFORT == "high"
    assert debates.EFFORT != DEFAULT_EFFORT


def test_max_tokens_is_set_well_above_the_library_default() -> None:
    """pydantic-ai defaults max_tokens to 4096, which truncates silently.

    Thinking counts against this budget on Sonnet 5, so `effort="high"` can spend
    it before any answer is written — and a findings model whose lists default to
    `()` turns the truncated response into a clean "found nothing". Two live
    candidacy runs died this way with `finish_reason: length`.
    """

    agent = _build()
    assert agent.model_settings is not None
    assert agent.model_settings["max_tokens"] == AGENT_MAX_TOKENS
    assert AGENT_MAX_TOKENS >= 32_000  # 4096 is the default this exists to beat
