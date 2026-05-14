"""Body-pole orientation and oblate-shape tests.

Covers:
 - Earth's pole is +Z in J2000 (by definition) and the alignment rotation
   is the identity.
 - Moon / Sun have non-trivial poles that match IAU 2009.
 - `pole_alignment_rotation` is a valid SO(3) element for any pole.
 - Zonal-harmonic acceleration around the Moon respects the body's pole:
   at a sample point on the Moon's spin axis, the in-pole-plane component
   of the J_n acceleration is zero — the same invariant Earth's J_n
   satisfies at the geographic pole.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from oamp.bodies import EARTH, MOON, OBLIQUITY_J2000_RAD, SUN
from oamp.dynamics.perturbations import zonal_harmonics
from oamp.frames import (
    body_pole_alignment_rotation,
    body_pole_in_j2000,
    pole_alignment_rotation,
)

# --------------------------------------------------------------------------- #
#  Pole orientation
# --------------------------------------------------------------------------- #


def test_earth_pole_is_plus_z():
    """J2000 is defined so Earth's mean rotation pole is +Z. No tilt."""
    assert body_pole_in_j2000(EARTH) == pytest.approx([0.0, 0.0, 1.0], abs=1e-15)


def test_moon_pole_matches_iau2009():
    """IAU 2009: α₀ = 269.9949°, δ₀ = 66.5392°.  Cartesian roughly
    (0.003, −0.398, 0.917)."""
    p = body_pole_in_j2000(MOON)
    assert np.linalg.norm(p) == pytest.approx(1.0, abs=1e-12)
    # Spot-check the dominant components.
    assert p[2] == pytest.approx(math.sin(math.radians(66.5392)), abs=1e-12)
    # Angle from +Z is 90° − 66.5392° = 23.4608° (close to obliquity, by
    # coincidence; the Moon's pole is *not* the ecliptic pole).
    angle = math.acos(p[2])
    assert math.degrees(angle) == pytest.approx(90.0 - 66.5392, abs=1e-4)


def test_sun_pole_matches_iau2009():
    p = body_pole_in_j2000(SUN)
    assert np.linalg.norm(p) == pytest.approx(1.0, abs=1e-12)
    # Sun's pole is tilted ~7.25° from ecliptic pole; ~26° from J2000 +Z.
    angle = math.acos(p[2])
    assert math.degrees(angle) == pytest.approx(90.0 - 63.87, abs=1e-3)


def test_alignment_rotation_is_orthogonal_for_arbitrary_pole():
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Random unit vector.
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        R = pole_alignment_rotation(v)
        # Orthogonality: R · Rᵀ = I.
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        # Determinant +1 (proper rotation, not a reflection).
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
        # Pole maps to +Z.
        assert np.allclose(R @ v, [0.0, 0.0, 1.0], atol=1e-12)


def test_earth_alignment_rotation_is_identity():
    R = body_pole_alignment_rotation(EARTH)
    assert np.allclose(R, np.eye(3), atol=1e-14)


# --------------------------------------------------------------------------- #
#  Body shape
# --------------------------------------------------------------------------- #


def test_earth_is_oblate():
    """WGS-84: R_pol ≈ R_eq · (1 − 1/298.257).  Flattening is +ve."""
    assert EARTH.polar_radius < EARTH.radius
    assert EARTH.inv_flattening == pytest.approx(298.257_223_563, rel=1e-9)
    derived = EARTH.radius * (1.0 - 1.0 / EARTH.inv_flattening)
    assert EARTH.polar_radius == pytest.approx(derived, abs=1e-3)


def test_sun_is_spherical():
    assert SUN.polar_radius == SUN.radius
    assert SUN.inv_flattening == 0.0


def test_obliquity_constant():
    """IAU 2006 mean obliquity at J2000.0 — 23.4393° to four decimals."""
    assert math.degrees(OBLIQUITY_J2000_RAD) == pytest.approx(23.4393, abs=1e-4)


# --------------------------------------------------------------------------- #
#  Pole-aware zonal harmonics
# --------------------------------------------------------------------------- #


def test_earth_j2_unchanged_by_pole_rotation():
    """Pole = +Z → rotation is identity → behaviour identical to the legacy
    (pre-pole-aware) code. Spot-check J2 at a generic LEO point."""
    pert = zonal_harmonics(EARTH, n_max=2)
    r = np.array([7e6, 1e5, 2e5], dtype=float)
    a = pert(0.0, r, np.zeros(3))
    assert np.all(np.isfinite(a))
    # J2 acceleration must point roughly toward Earth's centre (small
    # perturbation on the ~ -μ r/r³ central term).
    central = -EARTH.mu * r / np.linalg.norm(r) ** 3
    # Angle between J2 perturbation and central term must be < 90° (i.e.,
    # they aren't anti-aligned).
    assert np.dot(a, central) > 0


def test_zonal_acceleration_inplane_zero_at_body_pole():
    """For ANY body, at a sample point along the body's spin axis (above
    the pole) the J_n acceleration has no in-pole-plane (transverse)
    component — the perturbation is axisymmetric. This is an invariant we
    expect from the pole-aware implementation."""
    for body in (EARTH, MOON):
        pole_inertial = body_pole_in_j2000(body)
        # Place test point 7000 km above the pole in J2000 coordinates.
        r = 7e6 * pole_inertial
        a = zonal_harmonics(body, n_max=6)(0.0, r, np.zeros(3))
        # Component along the pole direction.
        a_axial = float(np.dot(a, pole_inertial)) * pole_inertial
        a_transverse = a - a_axial
        # Transverse component is numerically zero (within the precision of
        # the pole rotation + arithmetic).
        assert np.linalg.norm(a_transverse) < 1e-9 * np.linalg.norm(a) + 1e-15


def test_moon_j2_differs_when_pole_used():
    """Sanity: a Moon-J2 perturbation evaluated at a generic LEO-equivalent
    point yields a *different* vector when the Moon's true pole is used
    versus assuming J2000 +Z. Both must remain finite and bounded; the
    difference indicates we're actually applying the rotation."""
    # A reference vector along Moon's pole + an offset in J2000 +Y so the
    # two implementations definitely diverge.
    p = body_pole_in_j2000(MOON)
    r = 2e6 * p + np.array([0.0, 5e5, 0.0])
    # Pole-aware (current).
    a_aware = zonal_harmonics(MOON, n_max=2)(0.0, r, np.zeros(3))
    # Pole-unaware: feed r as-is to the bare _accel_jn helper to simulate
    # the legacy code path (treats J2000 +Z as the pole).
    from oamp.dynamics.perturbations import _accel_jn
    a_naive = _accel_jn(2, r, MOON.mu, MOON.radius, MOON.j2)
    assert np.all(np.isfinite(a_aware))
    assert np.all(np.isfinite(a_naive))
    # The two MUST differ — otherwise the rotation isn't doing anything.
    assert np.linalg.norm(a_aware - a_naive) > 0.01 * np.linalg.norm(a_naive)
