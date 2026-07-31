from typing import Literal

from pydantic import AliasChoices, Field

from predictelection.clients._base_config import ConfigBase


Effort = Literal["low", "medium", "high", "xhigh", "max"]
"""Anthropic's effort levels, mirroring pydantic-ai's AnthropicEffort.

Spelled out rather than imported so configuration stays free of the LLM client:
importing pydantic_ai from a settings module would drag the model machinery into
everything that merely reads config.
"""


class AnthropicConfig(ConfigBase):
    api_key: str = Field(
        min_length=1, validation_alias=AliasChoices("anthropic_api_key")
    )
    default_model: str = "claude-sonnet-5"
    effort: Effort | None = Field(
        default=None,
        validation_alias=AliasChoices("anthropic_effort"),
        description=(
            "How much thinking and tool work the model spends per request. "
            "Unset means each agent picks its own level; setting it is a "
            "deliberate override that applies to every agent, which is what "
            "makes a cost/quality sweep possible without touching code. Not "
            "every model takes every level — `xhigh` needs Sonnet 5 or Opus 4.7 "
            "and later, and a level the model does not support is a 400 on "
            "every request rather than a silent downgrade."
        ),
    )
