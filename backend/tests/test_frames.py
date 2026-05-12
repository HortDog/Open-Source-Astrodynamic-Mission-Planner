"""Tests for Phase-4 frame transforms.

Uses SPICE kernels (skipped if unavailable).  The key invariant is that the
inertial → synodic → inertial round trip is the identity to numerical
precision, and that the Earth–Moon line is along +x in the synodic frame.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("spiceypy", reason="SPICE not installed")
from oamp import spice  # noqa: E402

# Ensure kernels are loaded before running these tests.
_loaded = spice.furnsh_dir()
if not _loaded:
    pytest.skip(
        "no SPICE kernels in data/kernels — run `pixi run kernels` first",
        allow_module_level=True,
    )

from oamp.frames import (  # noqa: E402
    em_synodic_basis,
    em_synodic_to_inertial,
    inertial_to_em_synodic,
)
from oamp.timescales import utc_iso_to_et  # noqa: E402

_TEST_UTC = "2026-01-01T00:00:00"


def test_synodic_basis_places_moon_on_positive_x():
    """In the EM synodic frame the Moon must sit at (+d_em, 0, 0) relative to
    Earth — by construction of the rotation."""
    t = utc_iso_to_et(_TEST_UTC)
    b = em_synodic_basis(t)
    r_moon_eci, _ = spice.body_state("MOON", t, "EARTH", "J2000")
    r_moon_syn = b.R_eci_to_syn @ np.asarray(r_moon_eci)
    assert abs(r_moon_syn[0] - b.d_em_m) < 1e-3, f"x = {r_moon_syn[0]}"
    assert abs(r_moon_syn[1]) < 1e-3, f"y = {r_moon_syn[1]}"
    assert abs(r_moon_syn[2]) < 1e-3, f"z = {r_moon_syn[2]}"


def test_inertial_to_synodic_round_trip_is_identity():
    """A spacecraft state round-tripped inertial → synodic → inertial should
    differ from the original by less than 1 mm in position and 1 µm/s in
    velocity (well below SPICE's own precision)."""
    t = utc_iso_to_et(_TEST_UTC)
    r0 = np.array([7_000_000.0, 2_000_000.0, 500_000.0])
    v0 = np.array([-500.0, 7_000.0, 1_000.0])
    r_syn, v_syn = inertial_to_em_synodic(r0, v0, t)
    r1, v1 = em_synodic_to_inertial(r_syn, v_syn, t)
    assert np.max(np.abs(r1 - r0)) < 1e-3
    assert np.max(np.abs(v1 - v0)) < 1e-6


def test_moon_in_synodic_is_at_l1_distance_minus_mu():
    """The Moon should map to ((1 − μ) · d_em, 0, 0) in barycentric synodic
    coordinates (since the synodic frame origin is the barycenter)."""
    from oamp.dynamics.cr3bp import EM_MU

    t = utc_iso_to_et(_TEST_UTC)
    r_moon, v_moon = spice.body_state("MOON", t, "EARTH", "J2000")
    r_syn, _ = inertial_to_em_synodic(np.asarray(r_moon), np.asarray(v_moon), t)
    b = em_synodic_basis(t)
    expected_x = (1.0 - EM_MU) * b.d_em_m
    assert abs(r_syn[0] - expected_x) < 1.0, f"Moon x = {r_syn[0]} vs {expected_x}"
    assert abs(r_syn[1]) < 1.0
    assert abs(r_syn[2]) < 1.0


def test_earth_in_synodic_is_at_minus_mu_times_d():
    """Earth (at the origin in the input ECI frame) maps to (−μ · d_em, 0, 0)."""
    from oamp.dynamics.cr3bp import EM_MU

    t = utc_iso_to_et(_TEST_UTC)
    r_syn, _ = inertial_to_em_synodic(np.zeros(3), np.zeros(3), t)
    b = em_synodic_basis(t)
    assert abs(r_syn[0] - (-EM_MU * b.d_em_m)) < 1.0
    assert abs(r_syn[1]) < 1.0
    assert abs(r_syn[2]) < 1.0
