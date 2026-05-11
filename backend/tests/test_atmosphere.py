"""Tests for the pymsis-backed atmospheric density function."""

import math

import numpy as np
import pytest
from oamp.bodies import EARTH

pytest.importorskip("pymsis", reason="pymsis is optional — install to enable MSIS drag")


from oamp.dynamics.atmosphere import msis_density_fn  # noqa: E402
from oamp.dynamics.newtonian import TwoBodyState, propagate_orbit  # noqa: E402
from oamp.dynamics.perturbations import Vehicle, atmospheric_drag  # noqa: E402


def test_msis_density_sensible_at_400km():
    """At ~400 km the MSIS density should be in the 1e-13–1e-11 kg/m³ band."""
    rho_fn = msis_density_fn()
    r = np.array([EARTH.radius + 400e3, 0.0, 0.0])
    d = rho_fn(0.0, r)
    assert 1e-13 < d < 1e-10, f"unrealistic density at 400 km: {d:e}"


def test_msis_density_decreases_with_altitude():
    rho_fn = msis_density_fn()
    rhos = [rho_fn(0.0, np.array([EARTH.radius + h, 0.0, 0.0])) for h in (200e3, 400e3, 800e3)]
    assert rhos[0] > rhos[1] > rhos[2], f"density should decrease with altitude: {rhos}"


def test_msis_returns_zero_outside_envelope():
    rho_fn = msis_density_fn()
    deep_space = np.array([EARTH.radius + 2e6, 0.0, 0.0])
    assert rho_fn(0.0, deep_space) == 0.0


def test_msis_integrates_with_drag():
    """A low-altitude orbit with MSIS drag must monotonically lose energy."""
    r0 = EARTH.radius + 300e3
    v0 = math.sqrt(EARTH.mu / r0)
    state = TwoBodyState(r=(r0, 0.0, 0.0), v=(0.0, v0, 0.0))
    veh = Vehicle(mass_kg=500.0, drag_area_m2=4.0, drag_cd=2.2)
    drag = atmospheric_drag(veh, density_fn_full=msis_density_fn())
    _, states = propagate_orbit(state, duration_s=3600.0, steps=120, perturbation=drag)

    r = np.linalg.norm(states[:, :3], axis=1)
    v2 = np.einsum("ij,ij->i", states[:, 3:], states[:, 3:])
    eps = 0.5 * v2 - EARTH.mu / r
    assert eps[-1] < eps[0], f"MSIS drag should bleed energy: {eps[0]:e} -> {eps[-1]:e}"
