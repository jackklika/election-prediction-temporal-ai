from pathlib import Path
from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_PATH: Final[Path] = Path(__file__).resolve().parents[2] / ".env"


class ConfigBase(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",  # We share .env across all client configs
    )
