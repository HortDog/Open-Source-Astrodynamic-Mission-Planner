"""Tests for the symplectic integrator family."""

import math

import numpy as np
from oamp.bodies import EARTH
from oamp.dynamics.integrators import propagate_symplectic
from oamp.dynamics.newtonian import Maneuver, TwoBodyState, propagate_orbit


def _leo_state() -> TwoBodyState:
    r0 = EARTH.radius + 500e3
    v0 = math.sqrt(EARTH.mu / r0)
    return TwoBodyState(r=(r0, 0.0, 0.0), v=(0.0, v0, 0.0))


def _energy(states: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(states[:, :3], axis=1)
    v2 = np.einsum("ij,ij->i", states[:, 3:], states[:, 3:])
    return 0.5 * v2 - EARTH.mu / r


def test_verlet_closes_one_orbit():
    """Verlet propagation closes a circular orbit to ~km precision (it's 2nd order)."""
    state = _leo_state()
    period = 2 * math.pi * math.sqrt((EARTH.radius + 500e3) ** 3 / EARTH.mu)
    _, states = propagate_symplectic(state, period, steps=200, method="verlet")
    np.testing.assert_allclose(states[-1, :3], states[0, :3], atol=2e3)
    np.testing.assert_allclose(states[-1, 3:], states[0, 3:], atol=2.0)


def test_yoshida4_higher_accuracy_than_verlet():
    """Yoshida-4 must achieve smaller closure error than Verlet at the same step count."""
    state = _leo_state()
    period = 2 * math.pi * math.sqrt((EARTH.radius + 500e3) ** 3 / EARTH.mu)
    _, s_verlet = propagate_symplectic(state, period, steps=200, method="verlet")
    _, s_y4 = propagate_symplectic(state, period, steps=200, method="yoshida4")
    err_verlet = float(np.linalg.norm(s_verlet[-1, :3] - s_verlet[0, :3]))
    err_y4 = float(np.linalg.norm(s_y4[-1, :3] - s_y4[0, :3]))
    assert err_y4 < err_verlet, f"Yoshida4 ({err_y4:.2e}) not better than Verlet ({err_verlet:.2e})"


def test_yoshida4_energy_drift_bounded():
    """Symplectic Yoshida-4 must bound energy drift over many orbits — DOP853
    actually loses energy monotonically, while Yoshida should oscillate around it."""
    state = _leo_state()
    period = 2 * math.pi * math.sqrt((EARTH.radius + 500e3) ** 3 / EARTH.mu)
    _, states = propagate_symplectic(state, 20 * period, steps=4000, method="yoshida4")
    eps = _energy(states)
    rel_drift = float((eps.max() - eps.min()) / abs(eps[0]))
    # Energy oscillates but stays within machine-precision of the initial.
    assert rel_drift < 1e-8, f"yoshida4 energy drift too large: {rel_drift:.2e}"


def test_symplectic_matches_dop853_short_arc():
    """For short arcs, symplectic and DOP853 must agree closely."""
    state = _leo_state()
    period = 2 * math.pi * math.sqrt((EARTH.radius + 500e3) ** 3 / EARTH.mu)
    _, sym = propagate_symplectic(state, period / 4, steps=200, method="yoshida4")
    _, ref = propagate_orbit(state, period / 4, steps=200)
    np.testing.assert_allclose(sym[-1, :3], ref[-1, :3], atol=100.0)


def test_symplectic_supports_maneuvers():
    """A prograde Δv must raise the apoapsis under the symplectic propagator too."""
    state = _leo_state()
    mans = [Maneuver(t_offset_s=60.0, dv_ric=(0.0, 100.0, 0.0))]
    _, states = propagate_symplectic(
        state,
        duration_s=3600.0,
        steps=300,
        method="yoshida4",
        maneuvers=mans,
    )
    r_final = float(np.linalg.norm(states[-1, :3]))
    v_final = float(np.linalg.norm(states[-1, 3:]))
    energy = 0.5 * v_final**2 - EARTH.mu / r_final
    a_new = -EARTH.mu / (2 * energy)
    assert a_new > (EARTH.radius + 500e3) + 50e3


def test_unknown_method_rejected():
    state = _leo_state()
    try:
        propagate_symplectic(state, 100.0, 10, method="rk4_classical")
    except ValueError as e:
        assert "unknown integrator" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown method")
