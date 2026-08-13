"""The geocoding client, with the Census service stubbed.

No network: the transport is mocked, the same way the workflow tests stub the
model. A suite that geocodes for real is one that fails offline, rate-limits
itself in CI, and takes minutes.

What is worth testing here is almost entirely about *not lying*. A geocoder's
failure modes are quiet — a swapped coordinate pair, a row matched to the wrong
input, a tie stored as a location — and every one of them produces a map that
looks fine and is wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
import csv
import io

import httpx
import pytest

from predictelection.clients.geocoding import CensusGeocoderConfig
from predictelection.geocoding import (
    Address,
    CensusBatchGeocoder,
    GeocodingError,
    MatchQuality,
    Point,
)


MATCHED = (
    "1600 Pennsylvania Ave NW, Washington, DC, 20500",
    "Match",
    "Exact",
    "1600 PENNSYLVANIA AVE NW, WASHINGTON, DC, 20500",
    "-77.0365,38.8977",  # lon,lat — the service emits x,y
    "76225813",
    "L",
    "11",
    "001",
    "980000",
    "1034",
)


def _response(rows: Sequence[Sequence[str]]) -> str:
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    return buffer.getvalue()


def _geocoder(handler, **config) -> CensusBatchGeocoder:
    return CensusBatchGeocoder(
        CensusGeocoderConfig(backoff_seconds=0, **config),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _serving(body: str, status: int = 200):
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    return handle


def test_a_match_carries_the_point_and_the_geography() -> None:
    geocoder = _geocoder(_serving(_response([("abc", *MATCHED)])))

    (result,) = geocoder.geocode(
        [Address(key="abc", street="1600 Pennsylvania Ave NW", city="Washington")]
    )

    assert result.key == "abc"
    assert result.quality is MatchQuality.EXACT
    assert result.point == Point(longitude=-77.0365, latitude=38.8977)
    assert result.geography.geoid == "110019800001034"
    assert result.source.startswith("census:")


def test_longitude_comes_first() -> None:
    """The bug this exists to prevent raises nothing and is invisible on a map
    until someone notices Wisconsin is in the Indian Ocean."""

    geocoder = _geocoder(_serving(_response([("abc", *MATCHED)])))
    (result,) = geocoder.geocode([Address(key="abc", street="somewhere")])

    assert result.point is not None
    assert result.point.longitude < 0, "US longitudes are negative"
    assert 0 < result.point.latitude < 90


def test_results_are_returned_in_input_order_not_response_order() -> None:
    """Batch geocoders do not preserve order, and position is not identity.

    Lining answers up by position rather than by key is how one donor's money
    ends up attributed to another donor's neighbourhood — silently, and in a way
    no downstream check would catch.
    """

    rows = [("third", *MATCHED), ("first", *MATCHED), ("second", *MATCHED)]
    geocoder = _geocoder(_serving(_response(rows)))

    results = geocoder.geocode(
        [
            Address(key="first", street="a"),
            Address(key="second", street="b"),
            Address(key="third", street="c"),
        ]
    )

    assert [result.key for result in results] == ["first", "second", "third"]


def test_an_address_the_service_ignored_still_gets_a_result() -> None:
    """One result per input, always. A dropped row that vanishes becomes an
    invisible hole in a density map — the failure a map cannot show you."""

    geocoder = _geocoder(_serving(_response([("first", *MATCHED)])))

    results = geocoder.geocode(
        [Address(key="first", street="a"), Address(key="second", street="b")]
    )

    assert len(results) == 2
    assert results[1].key == "second"
    assert results[1].quality is MatchQuality.NO_MATCH
    assert results[1].point is None


def test_a_tie_is_not_given_coordinates() -> None:
    """Several equally good matches is an answer, and it is not a location.
    Picking one would be a guess wearing the clothes of a measurement."""

    rows = [("abc", "123 Main St", "Tie", "", "", "", "", "", "", "", "", "")]
    geocoder = _geocoder(_serving(_response(rows)))

    (result,) = geocoder.geocode([Address(key="abc", street="123 Main St")])

    assert result.quality is MatchQuality.TIE
    assert result.point is None


def test_an_interpolated_match_is_not_called_exact() -> None:
    """Non_Exact is block-accurate, not building-accurate. Flattening the
    distinction would overstate precision on every map drawn from it."""

    rows = [("abc", *(MATCHED[:2] + ("Non_Exact",) + MATCHED[3:]))]
    geocoder = _geocoder(_serving(_response(rows)))

    (result,) = geocoder.geocode([Address(key="abc", street="a")])

    assert result.quality is MatchQuality.APPROXIMATE
    assert result.quality.located


def test_an_unparseable_row_costs_one_address_not_the_batch() -> None:
    """The response format is positional and unversioned, so this will happen."""

    rows = [("abc", *MATCHED), ("",), ("def", *MATCHED)]
    geocoder = _geocoder(_serving(_response(rows)))

    results = geocoder.geocode(
        [Address(key="abc", street="a"), Address(key="def", street="b")]
    )

    assert [result.quality for result in results] == [
        MatchQuality.EXACT,
        MatchQuality.EXACT,
    ]


def test_a_commas_in_an_address_do_not_shift_the_columns() -> None:
    """Written with csv.writer for this reason: "Apt 3, Building B" joined by
    hand would push the ZIP into the state column and match confidently wrong."""

    sent: dict[str, str] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        sent["body"] = request.content.decode()
        return httpx.Response(200, text=_response([("abc", *MATCHED)]))

    _geocoder(handle).geocode(
        [
            Address(
                key="abc",
                street="1 Main St, Apt 3",
                city="Madison",
                state="WI",
                postal_code="53703",
            )
        ]
    )

    _, _, after_header = sent["body"].partition('filename="addresses.csv"')
    _, _, uploaded = after_header.partition("\r\n\r\n")
    (row,) = csv.reader([uploaded.splitlines()[0]])
    assert row == ["abc", "1 Main St, Apt 3", "Madison", "WI", "53703"]


def test_a_server_error_raises_rather_than_reporting_misses() -> None:
    """ "The service was down" and "these addresses do not exist" must not look
    alike — one is worth retrying and the other is a fact about the data."""

    geocoder = _geocoder(_serving("upstream exploded", status=503), max_attempts=2)

    with pytest.raises(GeocodingError, match="after 2 attempts"):
        geocoder.geocode([Address(key="abc", street="a")])


def test_a_rejected_request_is_not_retried() -> None:
    """A 4xx will be a 4xx again, and each attempt costs minutes on a service
    this slow."""

    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="bad benchmark")

    with pytest.raises(GeocodingError, match="rejected"):
        _geocoder(handle, max_attempts=3).geocode([Address(key="abc", street="a")])
    assert attempts == 1


def test_the_source_names_the_vintage() -> None:
    """Block boundaries move between vintages, so a stored GEOID without one
    cannot be compared to anything or recomputed."""

    geocoder = _geocoder(
        _serving(_response([])),
        benchmark="Public_AR_Census2020",
        vintage="Census2020_Census2020",
    )
    assert geocoder.source == "census:Public_AR_Census2020/Census2020_Census2020"


def test_an_empty_batch_asks_nothing() -> None:
    def handle(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should not have been called")

    assert _geocoder(handle).geocode([]) == ()


def test_an_address_without_a_street_is_refused_at_construction() -> None:
    """Before a batch of ten thousand is uploaded and takes four minutes."""

    with pytest.raises(ValueError, match="no street line"):
        Address(key="abc", street="   ")
    with pytest.raises(ValueError, match="needs a key"):
        Address(key="", street="1 Main St")


def test_a_point_outside_the_world_is_refused() -> None:
    with pytest.raises(ValueError, match="longitude"):
        Point(longitude=-181, latitude=0)
    with pytest.raises(ValueError, match="latitude"):
        Point(longitude=0, latitude=91)
