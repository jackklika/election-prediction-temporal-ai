from pathlib import Path
from typing import Any, Final

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

_ENV_PATH: Final[Path] = Path(__file__).resolve().parents[2] / ".env"


class ConfigBase(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",  # We share .env across all client configs
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[Any, ...]:
        """Treat a blank environment variable as absent rather than as a value.

        This was added because ANTHROPIC_API_KEY is set to `""` in Claude Code
        and other environments. So for all configuration, we try to find the
        first non-blank string.
        """

        def non_blank_env() -> dict[str, Any]:
            return {
                name: value for name, value in env_settings().items() if value != ""
            }

        return (init_settings, non_blank_env, dotenv_settings, file_secret_settings)
