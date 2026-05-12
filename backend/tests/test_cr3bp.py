"""Tests for the CR3BP propagator, Lagrange points, and Jacobi constant."""

from __future__ import annotations

import numpy as np
import pytest
from oamp.dynamics.cr3bp import (
    EM_MU,
    compute_manifold,
    cr3bp_jacobian,
    cr3bp_rhs,
    find_planar_lyapunov,
    jacobi_constant,
    lagrange_points,
    manifold_eigendirections,
    monodromy_matrix,
    propagate_cr3bp,
    stm_rhs,
    wsb_capture_grid,
)

# --------------------------------------------------------------------------- #
#  Lagrange points
# --------------------------------------------------------------------------- #


def test_em_lagrange_points_match_published_values():
    """Earth–Moon L1..L3 values from Szebehely (1967), L4/L5 closed-form."""
    L = lagrange_points(EM_MU)
    # Published x-coordinates (non-dim, EM frame, primaries at −μ and 1−μ).
    assert abs(L.L1[0] - 0.836915) < 1e-4, f"L1_x={L.L1[0]}"
    assert abs(L.L2[0] - 1.155682) < 1e-4, f"L2_x={L.L2[0]}"
    assert abs(L.L3[0] - (-1.005063)) < 1e-4, f"L3_x={L.L3[0]}"
    # L4 / L5 are at the equilateral-triangle vertices.
    sqrt3_2 = np.sqrt(3.0) / 2.0
    assert abs(L.L4[0] - (0.5 - EM_MU)) < 1e-12
    assert abs(L.L4[1] - sqrt3_2) < 1e-12
    assert abs(L.L5[1] - (-sqrt3_2)) < 1e-12


def test_collinear_points_are_equilibria():
    """At the collinear Lagrange points, the CR3BP acceleration must vanish."""
    L = lagrange_points(EM_MU)
    for name, pt in (("L1", L.L1), ("L2", L.L2), ("L3", L.L3)):
        state = np.array([pt[0], pt[1], pt[2], 0.0, 0.0, 0.0])
        a = cr3bp_rhs(0.0, state, EM_MU)
        # Velocity components copy through; acceleration components must vanish.
        assert np.allclose(a[3:], 0.0, atol=1e-9), f"{name} not in equilibrium: a={a[3:]}"


def test_triangular_points_are_equilibria():
    L = lagrange_points(EM_MU)
    for name, pt in (("L4", L.L4), ("L5", L.L5)):
        state = np.array([pt[0], pt[1], pt[2], 0.0, 0.0, 0.0])
        a = cr3bp_rhs(0.0, state, EM_MU)
        assert np.allclose(a[3:], 0.0, atol=1e-12), f"{name} not in equilibrium: a={a[3:]}"


def test_lagrange_rejects_invalid_mu():
    with pytest.raises(ValueError):
        lagrange_points(0.0)
    with pytest.raises(ValueError):
        lagrange_points(0.5)
    with pytest.raises(ValueError):
        lagrange_points(-0.1)


# --------------------------------------------------------------------------- #
#  Jacobi constant
# --------------------------------------------------------------------------- #


def test_jacobi_constant_is_conserved_along_a_trajectory():
    """Integrate from a state with a non-trivial trajectory and check that
    the Jacobi constant drifts by less than 1e-10 over the run."""
    # Initial state near (but not at) L1, with a small kick.
    L = lagrange_points(EM_MU)
    state0 = np.array([L.L1[0] - 0.005, 0.0, 0.0, 0.0, 0.01, 0.0])
    C0 = jacobi_constant(state0, EM_MU)
    _, states = propagate_cr3bp(state0, (0.0, 3.0), EM_MU, steps=200)
    Cs = np.array([jacobi_constant(s, EM_MU) for s in states])
    drift = float(np.max(np.abs(Cs - C0)))
    assert drift < 1e-10, f"Jacobi drift {drift:e} exceeds budget"


def test_jacobi_at_lagrange_points_matches_potential():
    """At a stationary point (v=0), C = 2 U."""
    L = lagrange_points(EM_MU)
    one_minus_mu = 1.0 - EM_MU
    for pt in (L.L1, L.L4):
        x, y, z = pt
        r1 = np.sqrt((x + EM_MU) ** 2 + y * y + z * z)
        r2 = np.sqrt((x - one_minus_mu) ** 2 + y * y + z * z)
        u = 0.5 * (x * x + y * y) + one_minus_mu / r1 + EM_MU / r2 + 0.5 * EM_MU * one_minus_mu
        C = jacobi_constant(np.array([x, y, z, 0.0, 0.0, 0.0]), EM_MU)
        assert abs(C - 2.0 * u) < 1e-12


# --------------------------------------------------------------------------- #
#  Propagator basics
# --------------------------------------------------------------------------- #


def test_propagate_rejects_bad_shape():
    with pytest.raises(ValueError):
        propagate_cr3bp(np.zeros(3), (0.0, 1.0), EM_MU)


def test_propagate_is_deterministic():
    state0 = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
    t1, s1 = propagate_cr3bp(state0, (0.0, 0.5), EM_MU, steps=50)
    t2, s2 = propagate_cr3bp(state0, (0.0, 0.5), EM_MU, steps=50)
    assert np.allclose(t1, t2)
    assert np.allclose(s1, s2)


def test_l4_neighborhood_stays_bounded_for_em_mu():
    """For μ < μ_Routh ≈ 0.0385, L4 is linearly stable.  A small perturbation
    must remain bounded over many orbital periods.  EM_MU ≈ 0.0121 satisfies
    this, so a trajectory starting near L4 should not escape."""
    L = lagrange_points(EM_MU)
    # Start L4 with a tiny (1e-4) displacement.
    state0 = np.array([L.L4[0] + 1e-4, L.L4[1], 0.0, 0.0, 0.0, 0.0])
    _, states = propagate_cr3bp(state0, (0.0, 30.0), EM_MU, steps=400)
    max_dist = float(np.max(np.linalg.norm(states[:, :3] - np.array(L.L4), axis=1)))
    # Linear theory predicts oscillation up to ~10× the initial perturbation
    # for short integrations; pick a generous bound.
    assert max_dist < 0.05, f"L4 trajectory escaped: max distance = {max_dist}"


# --------------------------------------------------------------------------- #
#  Variational equations + STM
# --------------------------------------------------------------------------- #


def test_jacobian_matches_finite_difference():
    """Analytical CR3BP Jacobian must agree with a central-difference estimate."""
    state = np.array([0.7, 0.1, 0.05, 0.02, 0.03, -0.01])
    A_an = cr3bp_jacobian(state, EM_MU)
    eps = 1e-7
    A_fd = np.zeros((6, 6))
    for k in range(6):
        ep = state.copy()
        ep[k] += eps
        em = state.copy()
        em[k] -= eps
        A_fd[:, k] = (cr3bp_rhs(0.0, ep, EM_MU) - cr3bp_rhs(0.0, em, EM_MU)) / (2 * eps)
    assert np.allclose(A_an, A_fd, atol=1e-5), (
        f"max |Δ| = {np.max(np.abs(A_an - A_fd)):.3e}"
    )


def test_stm_at_origin_is_identity():
    """At t=0 the state transition matrix must be the identity."""
    state0 = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
    y0 = np.concatenate([state0, np.eye(6).flatten()])
    dot = stm_rhs(0.0, y0, EM_MU)
    # The state derivative is the usual CR3BP RHS.
    assert np.allclose(dot[:6], cr3bp_rhs(0.0, state0, EM_MU))
    # The STM derivative at t=0 with Φ=I is just A.
    A = cr3bp_jacobian(state0, EM_MU)
    assert np.allclose(dot[6:].reshape(6, 6), A)


def test_monodromy_has_unit_eigenvalue_for_closed_orbit():
    """A closed periodic orbit's monodromy spectrum contains λ=1 (the orbit's
    own tangent direction, which is invariant under the period map)."""
    orbit = find_planar_lyapunov(L_point=1, Ax=0.008, mu=EM_MU)
    M = monodromy_matrix(np.asarray(orbit.state0), orbit.period, EM_MU)
    eigvals = np.linalg.eigvals(M)
    # Find the eigenvalue closest to +1.
    closest_to_one = min(abs(eigvals - 1.0))
    assert closest_to_one < 1e-4, f"min |λ−1| = {closest_to_one:.3e}"


# --------------------------------------------------------------------------- #
#  Planar Lyapunov DC
# --------------------------------------------------------------------------- #


def test_lyapunov_dc_converges_and_closes_the_orbit():
    """The DC residual must drop below the tolerance, and integrating the
    converged IC for one full period must return to (almost) the same state."""
    orbit = find_planar_lyapunov(L_point=1, Ax=0.01, mu=EM_MU)
    assert orbit.dc_residual < 1e-10, f"residual = {orbit.dc_residual:.3e}"
    assert orbit.family == "lyapunov_L1"
    # Period closure check.
    _, states = propagate_cr3bp(
        np.asarray(orbit.state0), (0.0, orbit.period), EM_MU, steps=200,
    )
    closure_err = float(np.linalg.norm(states[-1] - np.asarray(orbit.state0)))
    assert closure_err < 1e-6, f"closure error = {closure_err:.3e}"


def test_lyapunov_rejects_bad_inputs():
    with pytest.raises(ValueError):
        find_planar_lyapunov(L_point=3, Ax=0.01, mu=EM_MU)
    with pytest.raises(ValueError):
        find_planar_lyapunov(L_point=1, Ax=-0.01, mu=EM_MU)


def test_lyapunov_l2_distinct_from_l1():
    """L1 and L2 Lyapunov orbits at the same Ax must have different periods
    and Jacobi constants — they sit on opposite sides of the Moon."""
    o1 = find_planar_lyapunov(L_point=1, Ax=0.008, mu=EM_MU)
    o2 = find_planar_lyapunov(L_point=2, Ax=0.008, mu=EM_MU)
    assert abs(o1.period - o2.period) > 0.01
    assert abs(o1.jacobi - o2.jacobi) > 1e-4


# --------------------------------------------------------------------------- #
#  Invariant manifolds
# --------------------------------------------------------------------------- #


def test_manifold_eigendirections_yield_inverse_eigenvalues():
    """For a symplectic monodromy, real eigenvalues come in pairs (λ, 1/λ).
    Our extracted stable/unstable values must satisfy λ_u · λ_s ≈ 1."""
    orbit = find_planar_lyapunov(L_point=1, Ax=0.008, mu=EM_MU)
    M = monodromy_matrix(np.asarray(orbit.state0), orbit.period, EM_MU)
    _, _, lam_u, lam_s = manifold_eigendirections(M)
    product = lam_u * lam_s
    # Tight bound: symplectic pairs are exact in theory; numerical drift is small.
    assert abs(product - 1.0) < 1e-3, f"λu·λs = {product:.6f}"


def test_unstable_manifold_diverges_from_periodic_orbit():
    """A perturbation along the unstable eigenvector must grow in time.
    Compare the trajectory radius at the start vs the end."""
    orbit = find_planar_lyapunov(L_point=1, Ax=0.008, mu=EM_MU)
    tubes = compute_manifold(
        np.asarray(orbit.state0), orbit.period, EM_MU,
        direction="unstable", branch="+", n_samples=4, duration=4.0, steps=50,
    )
    assert len(tubes) > 0, "no manifold trajectories produced"
    # At least one trajectory should drift substantially from its seed.
    drifted = False
    for tube in tubes:
        d_start = np.linalg.norm(tube[0, :3] - np.asarray(orbit.state0[:3]))
        d_end = np.linalg.norm(tube[-1, :3] - np.asarray(orbit.state0[:3]))
        if d_end > 10 * d_start + 1e-3:
            drifted = True
            break
    assert drifted, "unstable manifold failed to diverge from the periodic orbit"


# --------------------------------------------------------------------------- #
#  WSB diagnostic (coarse smoke test — not a physics validation)
# --------------------------------------------------------------------------- #


def test_wsb_grid_returns_expected_shape_and_classes():
    """The diagnostic must return a 2D grid with values in {−1, 0, 1} and at
    least some classified points (not all failures)."""
    alts = np.array([100e3, 1_000e3])
    angles = np.linspace(0.0, np.pi, 4)
    grid = wsb_capture_grid(alts, angles, mu=EM_MU, duration=1.5)
    assert grid.shape == (2, 4)
    assert set(np.unique(grid)).issubset({-1, 0, 1})
    # At least half the points must integrate cleanly (no −1).
    fails = int((grid == -1).sum())
    assert fails < 4, f"too many integration failures: {fails}"
