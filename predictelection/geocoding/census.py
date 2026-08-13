"""The Census Bureau's batch geocoder.

Free, no key, US only, and the only geocoder that returns census block GEOIDs
directly — which is the whole reason to prefer it here over a commercial
service. A block GEOID joins to tracts, block groups, ACS demographics and
precinct crosswalks without a second lookup.

It takes a CSV upload and returns a CSV, one row per input, capped at
`MAX_BATCH` addresses per request. Larger inputs are chunked here so callers can
hand over a million addresses and let this file worry about it.

Three things about the service shape the code:

**It is slow and it is flaky.** A full batch can take minutes and 5xx responses
under load are routine, so the timeout is generous and failures retry with
backoff. A batch that ultimately fails raises rather than returning misses —
"the service was down" and "these addresses do not exist" must not look alike.

**The response is not ordered and can be short.** Rows come back in whatever
order the service finishes them, and it has been known to drop rows. Everything
is matched back by the id column, and anything unaccounted for becomes an
explicit NO_MATCH.

**Benchmark and vintage are versioned reference data, not settings.** They pick
which street file and which census geography the answers come from, they change,
and answers from different vintages are not comparable. Both are configurable
and both are written into `source` on every result.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import csv
import io
import logging
import time

import httpx

from predictelection.clients.geocoding import CensusGeocoderConfig
from predictelection.geocoding.base import (
    Address,
    CensusGeography,
    GeocodeResult,
    GeocodingError,
    MatchQuality,
    Point,
    unmatched,
)


logger = logging.getLogger(__name__)


MAX_BATCH = 10_000
"""Addresses per request, per the service's documented cap."""

_RETURN_TYPE = "geographies"
"""`geographies` rather than `locations`: same cost, and it adds the FIPS codes.
Asking for coordinates alone and looking geography up later would mean a second
pass over millions of rows to learn something the first pass could have said."""

_RESPONSE_COLUMNS = (
    "key",
    "input_address",
    "match",
    "match_type",
    "matched_address",
    "coordinates",
    "tiger_line_id",
    "tiger_side",
    "state_fips",
    "county_fips",
    "tract",
    "block",
)
"""The `geographies` response layout, which is headerless.

Positional and undocumented as a contract, which makes it the fragile part of
this file. `_parse_row` therefore reads defensively and treats a row it cannot
understand as a miss rather than raising — one malformed row should not fail a
batch of ten thousand.
"""


class CensusBatchGeocoder:
    """A `Geocoder` backed by the Census Bureau's batch endpoint."""

    def __init__(
        self,
        config: CensusGeocoderConfig | None = None,
        *,
        http: httpx.Client | None = None,
    ) -> None:
        self._config = config or CensusGeocoderConfig()
        # Injected for tests, the same way the agents take a stub model: this
        # class is otherwise untestable without hitting a public service, and a
        # test suite that geocodes for real is one that fails offline.
        self._http = http or httpx.Client(timeout=self._config.timeout_seconds)

    @property
    def source(self) -> str:
        return f"census:{self._config.benchmark}/{self._config.vintage}"

    def geocode(self, addresses: Sequence[Address]) -> tuple[GeocodeResult, ...]:
        """Look up a batch, chunking to the service's limit.

        Results come back in input order regardless of what order the service
        answered in, and every input gets exactly one result.
        """

        if not addresses:
            return ()

        found: dict[str, GeocodeResult] = {}
        for chunk in _chunks(addresses, MAX_BATCH):
            for result in self._one_batch(chunk):
                found[result.key] = result

        return tuple(
            found.get(
                address.key,
                unmatched([address.key], source=self.source)[0],
            )
            for address in addresses
        )

    # ----------------------------------------------------------------------

    def _one_batch(self, addresses: Sequence[Address]) -> tuple[GeocodeResult, ...]:
        payload = _as_csv(addresses)
        body = self._post_with_retries(payload, count=len(addresses))
        parsed = tuple(
            result
            for row in csv.reader(io.StringIO(body))
            if (result := self._parse_row(row)) is not None
        )

        answered = {result.key for result in parsed}
        missing = [address.key for address in addresses if address.key not in answered]
        if missing:
            # Not an error: the service does drop rows. But it is worth saying
            # out loud, because a caller comparing input to output counts would
            # otherwise see only that some addresses "did not match".
            logger.warning(
                "census geocoder returned no row for %d of %d addresses",
                len(missing),
                len(addresses),
            )
        return parsed + unmatched(missing, source=self.source)

    def _post_with_retries(self, payload: str, *, count: int) -> str:
        """POST the batch, retrying transport failures and 5xx.

        A 4xx is not retried: a malformed request will be malformed again, and
        repeating it wastes minutes per attempt on a service this slow.
        """

        last: Exception | None = None
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                response = self._http.post(
                    self._config.url,
                    data={
                        "benchmark": self._config.benchmark,
                        "vintage": self._config.vintage,
                        "returntype": _RETURN_TYPE,
                    },
                    files={"addressFile": ("addresses.csv", payload, "text/csv")},
                )
            except httpx.HTTPError as error:
                last = error
            else:
                if response.status_code < 400:
                    return response.text
                if response.status_code < 500:
                    raise GeocodingError(
                        f"census geocoder rejected a batch of {count}: "
                        f"{response.status_code} {response.text[:200]}"
                    )
                last = GeocodingError(
                    f"census geocoder returned {response.status_code}"
                )

            if attempt < self._config.max_attempts:
                delay = self._config.backoff_seconds * attempt
                logger.warning(
                    "census geocoder attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt,
                    self._config.max_attempts,
                    last,
                    delay,
                )
                time.sleep(delay)

        raise GeocodingError(
            f"census geocoder failed after {self._config.max_attempts} attempts"
        ) from last

    def _parse_row(self, row: Sequence[str]) -> GeocodeResult | None:
        """One response row, or None if it is not one.

        Defensive by design: the response format is positional and unversioned,
        so a row that does not fit the expected shape is reported as a miss and
        logged rather than raising. Losing one address is recoverable; failing
        a ten-thousand-address batch on a stray line is not.
        """

        if len(row) < 3:
            return None

        fields = dict(zip(_RESPONSE_COLUMNS, row))
        key = fields.get("key", "").strip()
        if not key:
            return None

        match = fields.get("match", "").strip().lower()
        if match == "tie":
            return GeocodeResult(key=key, quality=MatchQuality.TIE, source=self.source)
        if match != "match":
            return GeocodeResult(
                key=key, quality=MatchQuality.NO_MATCH, source=self.source
            )

        point = _parse_coordinates(fields.get("coordinates", ""))
        if point is None:
            logger.warning("census geocoder matched %s with no coordinates", key)
            return GeocodeResult(
                key=key, quality=MatchQuality.NO_MATCH, source=self.source
            )

        return GeocodeResult(
            key=key,
            # The service says "Exact" or "Non_Exact"; anything it matched but
            # did not call exact is interpolated onto a street segment, which is
            # block-accurate and not building-accurate.
            quality=(
                MatchQuality.EXACT
                if fields.get("match_type", "").strip().lower() == "exact"
                else MatchQuality.APPROXIMATE
            ),
            source=self.source,
            point=point,
            matched_address=fields.get("matched_address") or None,
            geography=CensusGeography(
                state_fips=fields.get("state_fips") or None,
                county_fips=fields.get("county_fips") or None,
                tract=fields.get("tract") or None,
                block=fields.get("block") or None,
            ),
        )


# --------------------------------------------------------------------------


def _chunks(addresses: Sequence[Address], size: int) -> Iterator[Sequence[Address]]:
    for start in range(0, len(addresses), size):
        yield addresses[start : start + size]


def _as_csv(addresses: Sequence[Address]) -> str:
    """The service's input layout: id, street, city, state, zip. No header.

    Written with `csv.writer` rather than by joining commas, because addresses
    contain commas — "Apt 3, Building B" would otherwise shift every later
    column and produce confidently wrong matches instead of errors.
    """

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for address in addresses:
        writer.writerow(
            [
                address.key,
                address.street,
                address.city or "",
                address.state or "",
                address.postal_code or "",
            ]
        )
    return buffer.getvalue()


def _parse_coordinates(value: str) -> Point | None:
    """ "lon,lat" as the service writes it — longitude first.

    The one ordering worth a comment: the service emits x,y, so longitude leads.
    Reading it as lat,lon puts every address in the wrong hemisphere without
    raising anything, because most US longitudes are valid latitudes' negatives.
    """

    longitude, separator, latitude = value.partition(",")
    if not separator:
        return None
    try:
        return Point(longitude=float(longitude), latitude=float(latitude))
    except ValueError:
        return None


__all__ = ["MAX_BATCH", "CensusBatchGeocoder"]
