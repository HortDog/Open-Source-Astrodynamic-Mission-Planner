"""Tests for Hohmann and Lambert transfers."""

import math

import numpy as np
import pytest
from oamp.bodies import EARTH
from oamp.dynamics.newtonian import TwoBodyState, propagate_orbit
from oamp.dynamics.transfers import hohmann_transfer, lambert_universal

# --------------------------------------------------------------------------- #
#  Hohmann
# --------------------------------------------------------------------------- #


def test_hohmann_leo_to_geo():
    """Classic LEO (300 km) → GEO transfer. Curtis Example 6.1."""
    r1 = EARTH.radius + 300e3
    r2 = 42_164e3
    res = hohmann_transfer(r1, r2)
    # Expected ΔV_total ≈ 3.93 km/s, TOF ≈ 5.27 h.
    assert 3800 < res.dv_total_m_s < 4000
    assert 5.0 * 3600 < res.transfer_time_s < 5.5 * 3600
    # Δv1 > Δv2 for an apoapsis raise.
    assert res.dv1_m_s > res.dv2_m_s
    np.testing.assert_allclose(res.semi_major_axis_m, 0.5 * (r1 + r2))


def test_hohmann_reject_negative():
    with pytest.raises(ValueError):
        hohmann_transfer(-1.0, 1e7)


# --------------------------------------------------------------------------- #
#  Lambert  — round-trip via propagation
# --------------------------------------------------------------------------- #


def test_lambert_roundtrip_circular_quarter_orbit():
    """Pick two points on a known circular orbit, ask Lambert for the velocity,
    propagate forward, verify we arrive at the requested point."""
    r0 = EARTH.radius + 500e3
    v0 = math.sqrt(EARTH.mu / r0)
    period = 2 * math.pi * math.sqrt(r0**3 / EARTH.mu)
    tof = period / 4  # quarter orbit

    r1 = (r0, 0.0, 0.0)
    r2 = (0.0, r0, 0.0)

    res = lambert_universal(r1, r2, tof, mu=EARTH.mu)
    assert res.converged

    # The recovered v1 should be very close to (0, v0, 0).
    np.testing.assert_allclose(res.v1_m_s, (0.0, v0, 0.0), rtol=1e-4, atol=1.0)

    # Propagate and check we land near r2.
    state = TwoBodyState(r=r1, v=res.v1_m_s)
    _, traj = propagate_orbit(state, duration_s=tof, steps=200)
    np.testing.assert_allclose(traj[-1, :3], r2, atol=10.0)


def test_lambert_rejects_degenerate_180():
    """The 180° transfer has an undefined transfer plane and must be rejected."""
    r1 = EARTH.radius + 300e3
    r2 = EARTH.radius + 1500e3
    with pytest.raises(ValueError, match="180"):
        lambert_universal((r1, 0.0, 0.0), (-r2, 0.0, 0.0), 3000.0)


def test_hohmann_full_sequence_lands_on_target_orbit():
    """Run the full Hohmann sequence as propagated by the engine. Final state
    must be on a near-circular orbit at the target radius."""
    from oamp.dynamics.newtonian import Maneuver, propagate_orbit

    r1 = EARTH.radius + 400e3
    r2 = 42_164e3
    ho = hohmann_transfer(r1, r2)
    T_leo = 2 * math.pi * math.sqrt(r1**3 / EARTH.mu)
    v_leo = math.sqrt(EARTH.mu / r1)

    _, states = propagate_orbit(
        TwoBodyState(r=(r1, 0.0, 0.0), v=(0.0, v_leo, 0.0)),
        duration_s=T_leo + ho.transfer_time_s + 100.0,
        steps=400,
        maneuvers=[
            Maneuver(t_offset_s=T_leo, dv_ric=(0.0, ho.dv1_m_s, 0.0)),
            Maneuver(t_offset_s=T_leo + ho.transfer_time_s, dv_ric=(0.0, ho.dv2_m_s, 0.0)),
        ],
    )
    # Final state should be at apoapsis side, ~r2 radius, speed ≈ v_circ(r2).
    r_final = np.linalg.norm(states[-1, :3])
    v_final = np.linalg.norm(states[-1, 3:])
    v_circ = math.sqrt(EARTH.mu / r2)
    assert abs(r_final - r2) / r2 < 1e-3, f"final r={r_final:.0f}, target {r2:.0f}"
    assert abs(v_final - v_circ) / v_circ < 1e-3, f"v={v_final:.2f}, v_circ={v_circ:.2f}"


def test_lambert_arbitrary_geometry():
    """Pick a 120° transfer and verify Lambert + propagation closes the loop."""
    r1 = (EARTH.radius + 500e3, 0.0, 0.0)
    # 120° from r1 in the equatorial plane, slightly higher altitude.
    cos_a, sin_a = math.cos(math.radians(120)), math.sin(math.radians(120))
    r2_n = EARTH.radius + 800e3
    r2 = (r2_n * cos_a, r2_n * sin_a, 0.0)
    tof = 1500.0  # s

    res = lambert_universal(r1, r2, tof)
    assert res.converged

    # Propagate from r1 with v1 and check we arrive at r2.
    state = TwoBodyState(r=r1, v=res.v1_m_s)
    _, traj = propagate_orbit(state, duration_s=tof, steps=100)
    np.testing.assert_allclose(traj[-1, :3], r2, atol=100.0)
    # Returned v2 must match the propagated final velocity.
    np.testing.assert_allclose(traj[-1, 3:], res.v2_m_s, atol=0.5)
