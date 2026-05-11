"""Tests for TLE ingest and SGP4 propagation.

We use a frozen ISS TLE from a known epoch so the test is reproducible and
doesn't hit Celestrak (which would make CI flaky).
"""

import math

import pytest
from oamp.bodies import EARTH
from oamp.tle import jd_from_iso_utc, parse_tle, propagate_tle, tle_state

# ISS (ZARYA) NORAD 25544 — epoch 2024-01-01T00:00 UTC, mean elements only.
# Format strictly follows the NORAD spec (69 chars per line incl. checksum).
ISS_NAME = "ISS (ZARYA)"
ISS_L1 = "1 25544U 98067A   24001.50000000  .00012345  00000+0  22222-3 0  9991"
ISS_L2 = "2 25544  51.6400 100.0000 0001000  90.0000 270.0000 15.50000000123456"


def test_parse_tle_basic():
    sat = parse_tle(ISS_L1, ISS_L2, ISS_NAME)
    assert sat.satnum == 25544
    # Inclination from the TLE is 51.64° (in radians for the sgp4 lib).
    assert abs(math.degrees(sat.inclo) - 51.64) < 1e-6


def test_parse_tle_rejects_short_lines():
    with pytest.raises(ValueError):
        parse_tle("1 25544U", "2 25544", "short")


def test_parse_tle_rejects_wrong_prefix():
    bad1 = "X 25544U 98067A   24001.50000000  .00012345  00000+0  22222-3 0  9991"
    with pytest.raises(ValueError):
        parse_tle(bad1, ISS_L2)


def test_propagate_at_epoch_yields_leo_radius():
    """At the TLE epoch the spacecraft must sit at a sensible LEO radius."""
    state = tle_state(ISS_L1, ISS_L2, ISS_NAME, norad_id=25544)
    r_mag = math.sqrt(sum(x * x for x in state.r_m))
    v_mag = math.sqrt(sum(x * x for x in state.v_m_s))
    # ISS orbits at ~400-420 km altitude.
    assert 350e3 < r_mag - EARTH.radius < 500e3, f"altitude={r_mag - EARTH.radius:.0f}m"
    # Speed near circular LEO velocity.
    assert 7400 < v_mag < 7900, f"speed={v_mag:.0f} m/s"
    # Period from mean motion: 15.5 rev/day → ~92.9 minutes.
    assert 88 < state.period_minutes < 100


def test_propagate_forward_a_few_minutes_changes_position():
    """Propagating 10 minutes after epoch must change the spacecraft position
    by a sensible amount (close to v * t for LEO)."""
    sat = parse_tle(ISS_L1, ISS_L2, ISS_NAME)
    jd0, fr0 = sat.jdsatepoch, sat.jdsatepochF
    r0, v0, _ = propagate_tle(sat, jd0, fr0)

    # 10 minutes later — add to the fractional day component.
    fr1 = fr0 + 10.0 / 1440.0
    r1, _, _ = propagate_tle(sat, jd0, fr1)
    displacement = math.sqrt(sum((r1[i] - r0[i]) ** 2 for i in range(3)))
    # ISS travels ~4500-4700 km in 10 minutes along its orbit.
    assert 4_000e3 < displacement < 5_000e3, f"displacement={displacement:.0f}m"


def test_jd_from_iso_round_trip():
    """Converting a known epoch and propagating back should give consistent JD."""
    jd, fr = jd_from_iso_utc("2024-01-01T12:00:00")
    # JD for 2024-01-01 12:00 UTC ≈ 2_460_311.0
    total = jd + fr
    assert abs(total - 2_460_311.0) < 0.01
