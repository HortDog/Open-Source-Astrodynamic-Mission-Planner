"""Tests for the Greenwich-angle helper and the ECEF ↔ J2000 round trip.

The GMST model is a constant-rate approximation that mirrors the front-end
`web/src/render/gmst.ts` so the two sides compute the same angle for a given
TDB epoch — important for ECEF initial conditions submitted by the editor
to round-trip through the backend.

No SPICE required: ECEF is a pure Z-axis rotation by θ_G, plus the rotating-
frame velocity correction.
"""

from __future__ import annotations

import numpy as np
import pytest
from oamp.frames import (
    EARTH_OMEGA_RAD_S,
    EARTH_OMEGA_VEC,
    GMST_AT_J2000_RAD,
    ecef_to_j2000,
    gmst_from_tdb,
    j2000_to_ecef,
)

# --------------------------------------------------------------------------- #
#  GMST
# --------------------------------------------------------------------------- #


def test_gmst_at_j2000_is_the_constant():
    """At t = 0 TDB, GMST is the seeded J2000 angle (~18h 41m 50.5s)."""
    assert gmst_from_tdb(0.0) == pytest.approx(GMST_AT_J2000_RAD, abs=1e-12)


def test_gmst_advances_at_omega_earth():
    """One second of TDB advances GMST by ω⊕."""
    g0 = gmst_from_tdb(0.0)
    g1 = gmst_from_tdb(1.0)
    assert (g1 - g0) == pytest.approx(EARTH_OMEGA_RAD_S, abs=1e-15)


def test_gmst_wraps_after_one_sidereal_day():
    """After 86 164.0905 s (sidereal day) GMST returns near its starting value."""
    sidereal_day = 2.0 * np.pi / EARTH_OMEGA_RAD_S
    g0 = gmst_from_tdb(0.0)
    g1 = gmst_from_tdb(sidereal_day)
    assert (g1 - g0) == pytest.approx(0.0, abs=1e-9)


def test_gmst_handles_nan_safely():
    assert gmst_from_tdb(float("nan")) == 0.0


# --------------------------------------------------------------------------- #
#  ECEF ↔ J2000 round trip
# --------------------------------------------------------------------------- #


def test_ecef_round_trip_preserves_state():
    """Apply j2000→ecef then ecef→j2000; the result must match the input
    to floating-point precision at every epoch we test."""
    r = np.array([7e6, 1e5, -2e5], dtype=float)
    v = np.array([10.0, 7500.0, 200.0], dtype=float)
    for t_tdb in (0.0, 1.0, 3600.0, 86164.0, 1.5e8, -1.0e6):
        r_ecef, v_ecef = j2000_to_ecef(r, v, t_tdb)
        r_back, v_back = ecef_to_j2000(r_ecef, v_ecef, t_tdb)
        assert np.allclose(r_back, r, atol=1e-6, rtol=1e-12)
        assert np.allclose(v_back, v, atol=1e-9, rtol=1e-12)


def test_ecef_at_t_zero_is_a_z_rotation_by_gmst_j2000():
    """At t=0 the ECEF rotation is exactly R_z(−GMST_AT_J2000)."""
    r = np.array([1e7, 0.0, 0.0], dtype=float)
    v = np.zeros(3)
    r_ecef, _ = j2000_to_ecef(r, v, 0.0)
    theta = GMST_AT_J2000_RAD
    expected = np.array([
        1e7 * np.cos(theta),
        -1e7 * np.sin(theta),
        0.0,
    ])
    assert np.allclose(r_ecef, expected, rtol=1e-12)


def test_ecef_velocity_picks_up_minus_omega_cross_r():
    """For a stationary point in ECI, velocity in ECEF is −ω × r."""
    r = np.array([6.5e6, 1.2e6, 3.0e5], dtype=float)
    v_eci = np.zeros(3)
    _, v_ecef = j2000_to_ecef(r, v_eci, 0.0)
    # Forward path rotates v_eci=0 to 0, then subtracts ω × r_ecef.
    r_ecef, _ = j2000_to_ecef(r, v_eci, 0.0)
    expected = -np.cross(EARTH_OMEGA_VEC, r_ecef)
    assert np.allclose(v_ecef, expected, rtol=1e-12)


def test_z_axis_state_invariant_to_rotation():
    """A state along the rotation axis (pure z) is unchanged by ECEF rotation,
    and a velocity along +z stays along +z."""
    r = np.array([0.0, 0.0, 7e6], dtype=float)
    v = np.array([0.0, 0.0, 100.0], dtype=float)
    r_ecef, v_ecef = j2000_to_ecef(r, v, 12345.6)
    assert np.allclose(r_ecef, r, atol=1e-6)
    assert np.allclose(v_ecef, v, atol=1e-9)
