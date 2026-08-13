"""Addresses to points and census geographies.

    from predictelection.geocoding import CensusBatchGeocoder, Address

    geocoder = CensusBatchGeocoder()
    results = geocoder.geocode([Address(key="1", street="1600 Pennsylvania Ave NW",
                                       city="Washington", state="DC")])

One backend today. `postgis_tiger_geocoder` is already an extension in the
compose image with its tables created and empty, so the second backend is a
matter of loading TIGER shapefiles and writing a `Geocoder` that calls
`tiger.geocode` — no caller changes, which is what the protocol is for.

Nothing here touches the database. Storing a result, caching it by address, and
joining points to polygons are separate concerns from asking where an address
is; keeping them apart is what lets geocoding be a resumable second pass over
rows that are already imported.
"""

from __future__ import annotations

from predictelection.geocoding.base import (
    Address,
    CensusGeography,
    GeocodeResult,
    Geocoder,
    GeocodingError,
    MatchQuality,
    Point,
    unmatched,
)
from predictelection.geocoding.census import MAX_BATCH, CensusBatchGeocoder


__all__ = [
    "MAX_BATCH",
    "Address",
    "CensusBatchGeocoder",
    "CensusGeography",
    "GeocodeResult",
    "Geocoder",
    "GeocodingError",
    "MatchQuality",
    "Point",
    "unmatched",
]
