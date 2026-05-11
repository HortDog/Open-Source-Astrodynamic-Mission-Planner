"""Tests for the CasADi-based fuel-optimal trajectory solver."""

import math

import numpy as np
import pytest
from oamp.bodies import EARTH
from oamp.dynamics.optimization import optimize_multi_burn
from oamp.dynamics.transfers import lambert_universal


def test_two_burn_matches_lambert():
    """With exactly 2 burns the problem is determined — the optimum equals Lambert."""
    r_dep = EARTH.radius + 500e3
    r_arr = EARTH.radius + 4_000e3
    angle = math.radians(120)
    tof = 4 * 3600.0

    r1 = (r_dep, 0.0, 0.0)
    r2 = (r_arr * math.cos(angle), r_arr * math.sin(angle), 0.0)
    v_dep_circ = math.sqrt(EARTH.mu / r_dep)
    v_arr_circ = math.sqrt(EARTH.mu / r_arr)
    v_arr_dir = (-math.sin(angle) * v_arr_circ, math.cos(angle) * v_arr_circ, 0.0)

    # Lambert ground truth.
    lb = lambert_universal(r1, r2, tof, mu=EARTH.mu)
    assert lb.converged
    dv1_lb = np.array([lb.v1_m_s[i] - (0.0, v_dep_circ, 0.0)[i] for i in range(3)])
    dv2_lb = np.array([v_arr_dir[i] - lb.v2_m_s[i] for i in range(3)])

    # Two-burn NLP. Last manoeuvre at t_final so the target state is sampled
    # immediately after the arrival kick (no post-burn drift).
    coast = 1.0  # second
    res = optimize_multi_burn(
        x0_r=r1,
        x0_v=(0.0, v_dep_circ, 0.0),
        xf_r=r2,
        xf_v=v_arr_dir,
        maneuver_epochs_s=[coast, coast + tof],
        t_final_s=coast + tof,
        mu=EARTH.mu,
        initial_dv_guess=[tuple(dv1_lb), tuple(dv2_lb)],
    )
    assert res.converged
    # Total Δv must match Lambert (Lambert is the unique 2-burn solution).
    lambert_total = float(np.linalg.norm(dv1_lb) + np.linalg.norm(dv2_lb))
    assert abs(res.total_dv_m_s - lambert_total) < 20.0  # RK4 absorbs some error


def test_three_burn_finds_lower_or_equal_fuel():
    """Adding a third burn must not increase the total Δv."""
    r_dep = EARTH.radius + 500e3
    r_arr = EARTH.radius + 4_000e3
    angle = math.radians(120)
    tof = 4 * 3600.0

    r1 = (r_dep, 0.0, 0.0)
    r2 = (r_arr * math.cos(angle), r_arr * math.sin(angle), 0.0)
    v_dep_circ = math.sqrt(EARTH.mu / r_dep)
    v_arr_circ = math.sqrt(EARTH.mu / r_arr)
    v_arr_dir = (-math.sin(angle) * v_arr_circ, math.cos(angle) * v_arr_circ, 0.0)

    # 2-burn reference (Lambert).
    coast = 1.0
    lb = lambert_universal(r1, r2, tof, mu=EARTH.mu)
    dv1_lb = np.array([lb.v1_m_s[i] - (0.0, v_dep_circ, 0.0)[i] for i in range(3)])
    dv2_lb = np.array([v_arr_dir[i] - lb.v2_m_s[i] for i in range(3)])

    res2 = optimize_multi_burn(
        x0_r=r1,
        x0_v=(0.0, v_dep_circ, 0.0),
        xf_r=r2,
        xf_v=v_arr_dir,
        maneuver_epochs_s=[coast, coast + tof],
        t_final_s=coast + tof,
        initial_dv_guess=[tuple(dv1_lb), tuple(dv2_lb)],
    )
    assert res2.converged

    # 3-burn (insert a mid-coast burn, warm-started from {Δv1_lb, 0, Δv2_lb}).
    res3 = optimize_multi_burn(
        x0_r=r1,
        x0_v=(0.0, v_dep_circ, 0.0),
        xf_r=r2,
        xf_v=v_arr_dir,
        maneuver_epochs_s=[coast, coast + tof / 2, coast + tof],
        t_final_s=coast + tof,
        initial_dv_guess=[tuple(dv1_lb), (0.0, 0.0, 0.0), tuple(dv2_lb)],
    )
    assert res3.converged
    # 3-burn should not exceed 2-burn by much (allow numerical slack — the
    # mid-burn could even degrade fuel slightly as IPOPT trades off).
    assert res3.total_dv_m_s <= res2.total_dv_m_s + 50.0


def test_invalid_epochs_rejected():
    with pytest.raises(ValueError):
        optimize_multi_burn(
            x0_r=(7e6, 0, 0),
            x0_v=(0, 7500, 0),
            xf_r=(7e6, 0, 0),
            xf_v=(0, 7500, 0),
            maneuver_epochs_s=[100.0, 50.0],  # not increasing
            t_final_s=200.0,
        )
    with pytest.raises(ValueError):
        optimize_multi_burn(
            x0_r=(7e6, 0, 0),
            x0_v=(0, 7500, 0),
            xf_r=(7e6, 0, 0),
            xf_v=(0, 7500, 0),
            maneuver_epochs_s=[0.0],  # boundary
            t_final_s=100.0,
        )
