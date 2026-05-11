"""Tests for the perturbation module: zonal harmonics, drag, SRP.

Third-body tests are gated on SPICE availability since they need DE440s
loaded; we run them only when the kernels directory is populated.
"""

import math

import numpy as np
import pytest
from oamp.bodies import EARTH, SOLAR_RADIATION_PRESSURE_AU
from oamp.dynamics.newtonian import TwoBodyState, propagate_orbit
from oamp.dynamics.perturbations import (
    Vehicle,
    atmospheric_drag,
    compose,
    solar_radiation_pressure,
    zonal_harmonics,
)


def _leo_state() -> TwoBodyState:
    # 400 km circular polar orbit — exercises Earth oblateness without trivial symmetry.
    r0 = EARTH.radius + 400e3
    v0 = math.sqrt(EARTH.mu / r0)
    return TwoBodyState(r=(r0, 0.0, 0.0), v=(0.0, 0.0, v0))


# --------------------------------------------------------------------------- #
#  Zonal harmonics
# --------------------------------------------------------------------------- #


def test_zonal_j2_matches_legacy_path():
    """The new perturbation framework must reproduce the old j2_enabled path."""
    state = _leo_state()
    duration = 3600.0
    _, legacy = propagate_orbit(state, duration, steps=200, j2_enabled=True)
    _, viapert = propagate_orbit(
        state,
        duration,
        steps=200,
        perturbation=zonal_harmonics(EARTH, n_max=2),
    )
    np.testing.assert_allclose(legacy, viapert, rtol=1e-12, atol=1e-9)


def test_zonal_higher_order_adds_signal():
    """J3-J6 should produce a small but non-zero divergence from J2-only."""
    state = _leo_state()
    duration = 6 * 3600.0
    _, j2_only = propagate_orbit(
        state,
        duration,
        steps=400,
        perturbation=zonal_harmonics(EARTH, n_max=2),
    )
    _, full = propagate_orbit(
        state,
        duration,
        steps=400,
        perturbation=zonal_harmonics(EARTH, n_max=6),
    )
    diff = np.linalg.norm(full[:, :3] - j2_only[:, :3], axis=1)
    # J3 dominates the residual for a polar orbit; expect ~meters over 6 hours.
    assert diff.max() > 0.5, "higher zonals should produce >0.5 m position diff"
    assert diff.max() < 1.0e4, "should not blow up — bug if multi-km divergence"


def test_zonal_n_max_rejection():
    with pytest.raises(ValueError):
        zonal_harmonics(EARTH, n_max=1)
    with pytest.raises(ValueError):
        zonal_harmonics(EARTH, n_max=7)


# --------------------------------------------------------------------------- #
#  Atmospheric drag
# --------------------------------------------------------------------------- #


def test_drag_decreases_orbital_energy():
    """Drag must monotonically remove specific orbital energy."""
    # 250 km orbit — well within sensible drag regime.
    r0 = EARTH.radius + 250e3
    v0 = math.sqrt(EARTH.mu / r0)
    state = TwoBodyState(r=(r0, 0.0, 0.0), v=(0.0, v0, 0.0))
    veh = Vehicle(mass_kg=500.0, drag_area_m2=4.0, drag_cd=2.2)
    pert = atmospheric_drag(veh)
    _, states = propagate_orbit(state, duration_s=2 * 3600.0, steps=200, perturbation=pert)

    # Specific orbital energy ε = v²/2 − μ/r.
    r = np.linalg.norm(states[:, :3], axis=1)
    v2 = np.einsum("ij,ij->i", states[:, 3:], states[:, 3:])
    eps = 0.5 * v2 - EARTH.mu / r
    # Average energy must decrease over time (orbit slowly decays).
    assert eps[-1] < eps[0], "drag should bleed energy"
    # And the decay should be smooth (no spurious oscillations).
    assert np.all(np.diff(eps) <= 1e-6), "energy should be non-increasing"


def test_drag_vanishes_at_high_altitude():
    """At 1500 km the exponential model returns 0 → no acceleration."""
    veh = Vehicle(mass_kg=100.0)
    pert = atmospheric_drag(veh)
    high = np.array([EARTH.radius + 1.6e6, 0.0, 0.0])
    v = np.array([0.0, 7000.0, 0.0])
    a = pert(0.0, high, v)
    np.testing.assert_allclose(a, np.zeros(3))


# --------------------------------------------------------------------------- #
#  SRP (acceleration magnitude only — eclipse logic is tricky to unit-test
#  without SPICE; we just verify the order of magnitude in sunlight).
# --------------------------------------------------------------------------- #


@pytest.fixture
def _spice_loaded() -> bool:
    try:
        from oamp import spice

        spice.furnsh_dir()
        return True
    except Exception:
        return False


def test_srp_magnitude(_spice_loaded):
    if not _spice_loaded:
        pytest.skip("SPICE kernels not available")
    # 1 m² absorber, 100 kg, fully absorbing, near Earth.
    veh = Vehicle(mass_kg=100.0, srp_area_m2=1.0, srp_cr=1.0)
    pert = solar_radiation_pressure(veh)
    # Pick a state well outside Earth shadow (on the day side).
    # Sun roughly in +X at J2000 epoch; place satellite at +X+R_E to be in sunlight.
    r = np.array([EARTH.radius + 1.0e6, 0.0, 0.0])
    v = np.array([0.0, 7500.0, 0.0])
    a = pert(0.0, r, v)
    # P0 * A/m ≈ 4.56e-6 * 1/100 = 4.56e-8 m/s². Allow factor-2 for distance variation.
    mag = float(np.linalg.norm(a))
    assert SOLAR_RADIATION_PRESSURE_AU * 0.01 * 0.5 < mag < SOLAR_RADIATION_PRESSURE_AU * 0.01 * 2.0


# --------------------------------------------------------------------------- #
#  Compose
# --------------------------------------------------------------------------- #


def test_compose_sums_accelerations():
    """compose(a, b)(...) must equal a(...) + b(...) element-wise."""
    zonal = zonal_harmonics(EARTH, n_max=2)
    veh = Vehicle(mass_kg=500.0, drag_area_m2=4.0)
    drag = atmospheric_drag(veh)
    combined = compose(zonal, drag)
    r = np.array([EARTH.radius + 300e3, 0.0, 0.0])
    v = np.array([0.0, 7700.0, 0.0])
    expected = zonal(0.0, r, v) + drag(0.0, r, v)
    np.testing.assert_allclose(combined(0.0, r, v), expected)


# --------------------------------------------------------------------------- #
#  Maneuvers
# --------------------------------------------------------------------------- #


def test_in_track_maneuver_raises_apoapsis():
    """A prograde in-track Δv at periapsis raises the apoapsis."""
    from oamp.dynamics.newtonian import Maneuver

    r0 = EARTH.radius + 400e3
    v0 = math.sqrt(EARTH.mu / r0)
    state = TwoBodyState(r=(r0, 0.0, 0.0), v=(0.0, v0, 0.0))
    # 100 m/s prograde kick after 60 s.
    mans = [Maneuver(t_offset_s=60.0, dv_ric=(0.0, 100.0, 0.0))]
    _, states = propagate_orbit(
        state,
        duration_s=3600.0,
        steps=300,
        maneuvers=mans,
    )

    # Apoapsis radius from final orbit must exceed r0 by a clear margin.
    r_final = np.linalg.norm(states[-1, :3])
    v_final = np.linalg.norm(states[-1, 3:])
    energy = 0.5 * v_final**2 - EARTH.mu / r_final
    a_new = -EARTH.mu / (2 * energy)
    # New semi-major axis should be noticeably larger than r0.
    assert a_new > r0 + 50e3, f"Δv didn't raise the orbit: a={a_new:.0f}, r0={r0:.0f}"


def test_maneuver_after_horizon_rejected():
    from oamp.dynamics.newtonian import Maneuver

    state = _leo_state()
    with pytest.raises(ValueError):
        propagate_orbit(state, 600.0, 100, maneuvers=[Maneuver(t_offset_s=900.0, dv_ric=(0, 0, 0))])
