"""Turning a filed address into a point and the geographies containing it.

Shaped like `storage/`: a protocol plus swappable backends, because the choice
between them is an operational one rather than a modelling one. The Census
batch service needs nothing installed and is rate-limited and remote;
`postgis_tiger_geocoder` is already an extension in this image and needs tens of
gigabytes of TIGER shapefiles loaded, after which it is offline and unlimited.
Which is right depends on whether you are geocoding a thousand addresses or
twenty million, and that decision should not reach the caller.

Three properties are load-bearing:

**Batch is the unit.** Every backend takes a sequence and returns a sequence.
The Census service is only offered in batches of thousands, and a per-address
interface would make its natural shape the awkward one; a local geocoder loops
happily. Optimising for the remote case costs the local one nothing.

**A tie is not a match.** When more than one address matches, the Census service
says so, and storing either candidate's coordinates would invent precision the
data does not have. `MatchQuality` keeps ties and misses distinguishable from
hits, and `GeocodeResult.point` is None for both.

**Every result names what produced it.** Block boundaries change between
vintages, so "which census block is this address in" has no answer without
saying which vintage was asked. `source` travels with the result so a stored
coordinate can be re-derived, compared across vintages, or invalidated when a
backend is replaced — the same reason every claim in this project carries its
provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Address:
    """One address to look up, as it was filed.

    `key` is the caller's own identifier and is echoed back on the result. It
    exists because batch geocoders do not guarantee response order and may drop
    rows entirely, so position is not a safe way to line answers up with
    questions — a misalignment there silently attributes one donor's money to
    another donor's neighbourhood.
    """

    key: str
    street: str
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("an address needs a key to be matched back to")
        if not self.street.strip():
            raise ValueError(f"address {self.key} has no street line")

    @property
    def one_line(self) -> str:
        parts = [self.street, self.city, self.state, self.postal_code]
        return ", ".join(part.strip() for part in parts if part and part.strip())


class MatchQuality(StrEnum):
    """How much the geocoder is willing to claim about an answer."""

    EXACT = "exact"
    """One match, and the input matched a known address as written."""

    APPROXIMATE = "approximate"
    """One match, reached by interpolating along a street segment or by
    correcting the input. The point is on the right block, not on the building."""

    TIE = "tie"
    """Several equally good matches. Deliberately not a coordinate: picking one
    would be a guess wearing the clothes of a measurement."""

    NO_MATCH = "no_match"
    """Nothing matched. Kept as a result rather than dropped — a missing row in
    a density map is invisible, and invisible gaps are the way a map lies."""

    @property
    def located(self) -> bool:
        return self in {MatchQuality.EXACT, MatchQuality.APPROXIMATE}


@dataclass(frozen=True, slots=True)
class Point:
    """WGS 84, the coordinate system PostGIS SRID 4326 expects.

    Longitude first, matching GeoJSON and `ST_MakePoint`, and against the
    latitude-first habit of every mapping UI. Named fields rather than a tuple
    precisely because that ordering is the classic silent bug — a swapped pair
    puts Wisconsin in Somalia and raises nothing.
    """

    longitude: float
    latitude: float

    def __post_init__(self) -> None:
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"longitude out of range: {self.longitude}")
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"latitude out of range: {self.latitude}")

    @property
    def wkt(self) -> str:
        """For `ST_GeomFromText(..., 4326)`."""

        return f"POINT({self.longitude} {self.latitude})"


@dataclass(frozen=True, slots=True)
class CensusGeography:
    """The nesting FIPS codes a point falls in, when the backend reports them.

    Stored as the parts rather than as one GEOID because the parts are what
    joins: a tract GEOID is state+county+tract, a block GEOID appends block, and
    a caller aggregating to counties should not be re-slicing a string. `geoid`
    reassembles when a whole key is wanted.
    """

    state_fips: str | None = None
    county_fips: str | None = None
    tract: str | None = None
    block: str | None = None

    @property
    def geoid(self) -> str | None:
        """The longest GEOID the parts support, or None if there is no state."""

        if not self.state_fips:
            return None
        parts = [self.state_fips, self.county_fips, self.tract, self.block]
        found: list[str] = []
        for part in parts:
            if not part:
                break
            found.append(part)
        return "".join(found)


@dataclass(frozen=True, slots=True)
class GeocodeResult:
    """One answer, tied back to the question by `key`."""

    key: str
    quality: MatchQuality
    source: str
    """Which backend and reference vintage answered, e.g.
    `census:Public_AR_Current/Census2020_Current`. Not decoration: block
    boundaries move between vintages, so a stored geography without one cannot
    be compared to anything or recomputed."""

    point: Point | None = None
    matched_address: str | None = None
    """What the geocoder thinks it matched, which is often not what was asked.
    Worth storing: it is the only way to notice that "123 Main St" resolved to a
    different Main St in a different town."""

    geography: CensusGeography = CensusGeography()

    def __post_init__(self) -> None:
        if self.point is None and self.quality.located:
            raise ValueError(f"{self.quality} result for {self.key} has no point")
        if self.point is not None and not self.quality.located:
            raise ValueError(f"{self.quality} result for {self.key} carries a point")


@runtime_checkable
class Geocoder(Protocol):
    """The contract a geocoding backend has to satisfy."""

    @property
    def source(self) -> str:
        """Backend and vintage, recorded on every result it produces."""
        ...

    def geocode(self, addresses: Sequence[Address]) -> tuple[GeocodeResult, ...]:
        """Look up a batch.

        Returns one result per input, including misses, in input order. A
        backend that receives no answer for an address must still return a
        `NO_MATCH` for it: a caller counting rows in against rows out is how a
        silently truncated batch gets noticed.
        """
        ...


class GeocodingError(RuntimeError):
    """The backend could not be reached or answered unusably.

    Distinct from a no-match, which is an answer. Retrying this may help;
    retrying a no-match will not.
    """


def unmatched(keys: Sequence[str], *, source: str) -> tuple[GeocodeResult, ...]:
    """NO_MATCH results for keys a backend said nothing about.

    Shared because every backend needs it and the property it protects — one
    result per input, always — is easy to lose in an error path.
    """

    return tuple(
        GeocodeResult(key=key, quality=MatchQuality.NO_MATCH, source=source)
        for key in keys
    )


__all__ = [
    "Address",
    "CensusGeography",
    "GeocodeResult",
    "Geocoder",
    "GeocodingError",
    "MatchQuality",
    "Point",
    "unmatched",
]
