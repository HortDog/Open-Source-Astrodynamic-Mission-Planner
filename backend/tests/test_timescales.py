"""Time-scale conversion tests (fallback path; no SPICE LSK needed)."""

import pytest
from oamp.timescales import (
    LeapSecondError,
    et_to_utc_iso,
    utc_iso_to_et,
)


def test_j2000_epoch_round_trips_to_zero():
    """J2000 TDB epoch is 2000-01-01T11:58:55.816Z."""
    et = utc_iso_to_et("2000-01-01T11:58:55.816Z")
    assert abs(et) < 1e-3  # within a millisecond of zero


def test_round_trip_modern_date():
    iso = "2026-05-10T12:34:56Z"
    et = utc_iso_to_et(iso)
    iso_back = et_to_utc_iso(et)
    et_again = utc_iso_to_et(iso_back)
    assert abs(et - et_again) < 1e-3


def test_post_2017_leap_seconds_constant():
    """No leap second between 2017-01-01 and the table cutoff: any UTC delta
    in that window equals the corresponding ET delta exactly."""
    et_a = utc_iso_to_et("2020-06-15T00:00:00Z")
    et_b = utc_iso_to_et("2020-06-15T01:00:00Z")
    assert abs((et_b - et_a) - 3600.0) < 1e-6


def test_leap_second_boundary_2017():
    """The 2016-12-31 -> 2017-01-01 step inserted one leap second; an interval
    spanning it should be one second longer in ET than in pure UTC clock-time."""
    before = utc_iso_to_et("2016-12-31T23:59:59Z")
    after = utc_iso_to_et("2017-01-01T00:00:01Z")
    # Naive UTC delta would be 2 seconds; with the leap, ET sees 3 seconds.
    assert abs((after - before) - 3.0) < 1e-3


def test_out_of_range_dates_raise():
    with pytest.raises(LeapSecondError):
        utc_iso_to_et("1968-01-01T00:00:00Z")  # before TAI-UTC was defined
    with pytest.raises(LeapSecondError):
        utc_iso_to_et("2030-01-01T00:00:00Z")  # past the embedded table
