from pydantic import AliasChoices, Field

from predictelection.clients._base_config import ConfigBase


class AnthropicConfig(ConfigBase):
    api_key: str = Field(
        min_length=1, validation_alias=AliasChoices("anthropic_api_key")
    )
    default_model: str = "claude-sonnet-4-6"
