"""Geocoder configuration.

Here rather than in `geocoding/` for the same reason every other client config
lives here: reading settings should not import the machinery that uses them, so
a caller choosing a backend does not drag `httpx` along to find out which one.

Every field lists its own name alongside the environment variable in
`AliasChoices`, which is not decoration. `ConfigBase` sets `extra="ignore"` so
that one `.env` can serve every client, and the cost is that a keyword the model
does not recognise is *silently dropped* rather than rejected — so a field whose
only alias is the env var cannot be set in code at all, and
`CensusGeocoderConfig(max_attempts=2)` would quietly keep the default. That cost
an hour of "why is my test still retrying three times"; listing both names is
what makes the config injectable.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field

from predictelection.clients._base_config import ConfigBase


class CensusGeocoderConfig(ConfigBase):
    url: str = Field(
        default=("https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"),
        validation_alias=AliasChoices("census_geocoder_url", "url"),
    )

    benchmark: str = Field(
        default="Public_AR_Current",
        validation_alias=AliasChoices("census_geocoder_benchmark", "benchmark"),
        description=(
            "Which street reference file to match against. `Public_AR_Current` "
            "tracks the latest release, which means answers can move under you; "
            "pin a dated benchmark when a run has to be reproducible."
        ),
    )
    vintage: str = Field(
        default="Census2020_Current",
        validation_alias=AliasChoices("census_geocoder_vintage", "vintage"),
        description=(
            "Which census geography the returned FIPS codes describe. Block and "
            "tract boundaries are redrawn every decade, so a block GEOID means "
            "nothing without this — and codes from two vintages must never be "
            "compared or joined."
        ),
    )

    timeout_seconds: float = Field(
        default=600.0,
        gt=0,
        validation_alias=AliasChoices(
            "census_geocoder_timeout_seconds", "timeout_seconds"
        ),
        description=(
            "Generous on purpose: a full ten-thousand-address batch regularly "
            "takes several minutes, and a timeout that fires mid-batch costs "
            "the whole batch."
        ),
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        validation_alias=AliasChoices("census_geocoder_max_attempts", "max_attempts"),
    )
    backoff_seconds: float = Field(
        default=5.0,
        ge=0,
        validation_alias=AliasChoices(
            "census_geocoder_backoff_seconds", "backoff_seconds"
        ),
        description="Multiplied by the attempt number, so retries space out.",
    )
