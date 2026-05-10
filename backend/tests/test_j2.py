"""J2 perturbation tests.

The classic check: a nearly-circular LEO inclined at i=51.6° should regress
its right-ascension of ascending node (RAAN) at a known rate predictable
from J2 alone:

    Ω̇ ≈ -(3/2) n J2 (R_E / p)² cos(i)        (Vallado §9.4)

For our test orbit (a = 7000 km, e = 0, i = 51.6°) this works out to about
-4.5°/day. Propagating half a day and comparing the RAAN delta against the
analytic prediction is a sharp test of both the J2 acceleration and the
integrator.
"""

import math

import numpy as np
from oamp.bodies import EARTH
from oamp.dynamics.newtonian import TwoBodyState, propagate_orbit


def raan_of(state: np.ndarray) -> float:
    """Right ascension of ascending node from a Cartesian state, in radians."""
    r = state[:3]
    v = state[3:6]
    h = np.cross(r, v)
    # Ascending-node vector lies along z_hat × h (when h_z > 0).
    n = np.array([-h[1], h[0], 0.0])
    return math.atan2(n[1], n[0])


def test_j2_nodal_regression_matches_analytic():
    a = 7_000_000.0  # 622 km altitude
    inc = math.radians(51.6)
    v_circ = math.sqrt(EARTH.mu / a)

    state = TwoBodyState(
        r=(a, 0.0, 0.0),
        v=(0.0, v_circ * math.cos(inc), v_circ * math.sin(inc)),
    )

    # Half a day, ~8 orbits.
    half_day = 43_200.0
    _, states = propagate_orbit(state, half_day, steps=2000, body=EARTH, j2_enabled=True)

    raan_0 = raan_of(states[0])
    raan_f = raan_of(states[-1])
    delta_raan = (raan_f - raan_0 + math.pi) % (2 * math.pi) - math.pi

    # Analytic prediction.
    n = math.sqrt(EARTH.mu / a**3)
    expected_rate = -1.5 * n * EARTH.j2 * (EARTH.radius / a) ** 2 * math.cos(inc)
    expected_delta = expected_rate * half_day

    # 5% tolerance — this captures both the J2 sign/magnitude and that the
    # integrator preserves orbital elements over 8 orbits.
    rel_err = abs(delta_raan - expected_delta) / abs(expected_delta)
    assert rel_err < 0.05, (
        f"J2 nodal regression off: got {math.degrees(delta_raan):.3f}°, "
        f"expected {math.degrees(expected_delta):.3f}° (rel err {rel_err:.3%})"
    )


def test_j2_zero_disabled_recovers_two_body():
    """With j2_enabled=False, propagation must match pure two-body to high precision."""
    a = 7_000_000.0
    v_circ = math.sqrt(EARTH.mu / a)
    state = TwoBodyState(r=(a, 0.0, 0.0), v=(0.0, v_circ, 0.0))
    period = 2 * math.pi * math.sqrt(a**3 / EARTH.mu)

    _, no_j2 = propagate_orbit(state, period, steps=200, body=EARTH, j2_enabled=False)
    np.testing.assert_allclose(no_j2[-1, :3], no_j2[0, :3], rtol=0, atol=1.0)
