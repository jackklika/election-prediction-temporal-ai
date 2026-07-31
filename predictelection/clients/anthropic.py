from pydantic import AliasChoices, Field

from predictelection.clients._base_config import ConfigBase


class AnthropicConfig(ConfigBase):
    api_key: str = Field(validation_alias=AliasChoices("anthropic_api_key"))
    default_model: str = "claude-sonnet-4-6"
