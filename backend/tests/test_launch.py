"""Launch simulator sanity tests."""

import math

import numpy as np
from oamp.bodies import EARTH
from oamp.dynamics.launch import default_falcon9_like, simulate_launch


def test_default_launch_reaches_stable_leo():
    res = simulate_launch(default_falcon9_like())
    states = np.array(res.states)

    # Liftoff state at the pad.
    np.testing.assert_allclose(states[0, :3], [EARTH.radius, 0.0, 0.0], atol=1.0)
    np.testing.assert_allclose(states[0, 3:6], [0.0, 0.0, 0.0], atol=1e-9)

    # Ordering: burnout < circularization < end.
    assert 0 < res.burnout_index < res.circularization_index < len(states) - 1

    # Final orbit must be bound, near-circular, and well above the atmosphere.
    final_r = states[-1, :3]
    final_v = states[-1, 3:6]
    r = float(np.linalg.norm(final_r))
    v2 = float(np.dot(final_v, final_v))
    energy = 0.5 * v2 - EARTH.mu / r
    assert energy < 0, "final state is not bound"
    assert res.final_periapsis_km > 150, f"periapsis too low: {res.final_periapsis_km:.0f} km"
    assert abs(res.final_apoapsis_km - res.final_periapsis_km) < 5, "not circular"

    # Speed at the final sample should match circular orbital velocity within a percent.
    v_circ = math.sqrt(EARTH.mu / r)
    assert abs(res.final_speed_m_s - v_circ) / v_circ < 0.01

    # Circularization Δv is small relative to ascent Δv (we tuned the ascent
    # to leave a low-energy apsis raise).
    assert 0 < res.circularization_dv_m_s < 1_500


def test_launch_stays_in_equatorial_plane():
    """The 2D simulator should never put any mass off the launch plane (Z=0)."""
    res = simulate_launch(default_falcon9_like())
    z = np.array(res.states)[:, 2]
    vz = np.array(res.states)[:, 5]
    assert float(np.max(np.abs(z))) < 1.0, "out-of-plane drift"
    assert float(np.max(np.abs(vz))) < 1e-3
