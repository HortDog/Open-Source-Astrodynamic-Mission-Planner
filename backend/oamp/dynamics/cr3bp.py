"""Circular Restricted Three-Body Problem (CR3BP) dynamics.

The CR3BP describes the motion of a massless test particle under the gravity
of two primaries moving on circular orbits about their common barycenter,
expressed in the *synodic* (corotating) frame.  In non-dimensional units
(distance L = primary separation, time T = 1/mean-motion, mass M = sum of
primary masses) the equations of motion are:

    ẍ =  2ẏ + x − (1−μ)(x+μ)/r₁³ − μ(x−1+μ)/r₂³
    ÿ = −2ẋ + y − (1−μ) y    /r₁³ − μ y      /r₂³
    z̈ =        −(1−μ) z    /r₁³ − μ z      /r₂³

where μ = m₂/(m₁+m₂), r₁ = ‖(x+μ, y, z)‖ is the distance to the larger
primary at (−μ, 0, 0), and r₂ = ‖(x−1+μ, y, z)‖ is the distance to the
smaller primary at (1−μ, 0, 0).

The Jacobi constant
    C(x,y,z,ẋ,ẏ,ż) = 2 U(x,y,z) − (ẋ²+ẏ²+ż²)
with U = ½(x²+y²) + (1−μ)/r₁ + μ/r₂ + ½μ(1−μ)
is an integral of motion (conserved to numerical-integration accuracy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from oamp.bodies import EARTH, MOON, SUN

# --------------------------------------------------------------------------- #
#  Canonical mass ratios.  These are derived from oamp.bodies so they stay
#  consistent with the rest of the engine; pre-computed for convenience.
# --------------------------------------------------------------------------- #

#: Earth–Moon mass ratio μ = m_Moon / (m_Earth + m_Moon)
EM_MU: float = MOON.mu / (EARTH.mu + MOON.mu)

#: Sun–Earth mass ratio μ = m_Earth / (m_Sun + m_Earth)
SE_MU: float = EARTH.mu / (SUN.mu + EARTH.mu)

#: Earth–Moon mean separation (m).  Used as the length scale L when
#: dimensionalising.  Source: IAU mean lunar distance.
EM_LENGTH_M: float = 384_400_000.0

#: Earth–Moon mean motion (rad/s).  T = 1/EM_MEAN_MOTION is the time scale.
EM_MEAN_MOTION_RAD_S: float = ((EARTH.mu + MOON.mu) / EM_LENGTH_M**3) ** 0.5


# --------------------------------------------------------------------------- #
#  Right-hand side
# --------------------------------------------------------------------------- #


def cr3bp_rhs(_t: float, y: np.ndarray, mu: float) -> np.ndarray:
    """Non-dimensional CR3BP RHS for use with scipy `solve_ivp`.

    State `y = [x, y, z, ẋ, ẏ, ż]` in synodic non-dimensional coordinates.
    """
    x, yy, z, vx, vy, vz = y
    # Distances to the two primaries.
    dx1 = x + mu
    dx2 = x - (1.0 - mu)
    r1_3 = (dx1 * dx1 + yy * yy + z * z) ** 1.5
    r2_3 = (dx2 * dx2 + yy * yy + z * z) ** 1.5
    one_minus_mu = 1.0 - mu

    ax = 2.0 * vy + x - one_minus_mu * dx1 / r1_3 - mu * dx2 / r2_3
    ay = -2.0 * vx + yy - one_minus_mu * yy / r1_3 - mu * yy / r2_3
    az = -one_minus_mu * z / r1_3 - mu * z / r2_3
    return np.array([vx, vy, vz, ax, ay, az])


# --------------------------------------------------------------------------- #
#  Jacobi constant
# --------------------------------------------------------------------------- #


def jacobi_constant(state: np.ndarray, mu: float) -> float:
    """Compute the Jacobi constant C = 2U − v² (non-dimensional).

    Conserved along any solution of the CR3BP to within integrator tolerance;
    drift is a sensitive integration-error gauge.
    """
    x, yy, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu) ** 2 + yy * yy + z * z)
    r2 = np.sqrt((x - (1.0 - mu)) ** 2 + yy * yy + z * z)
    one_minus_mu = 1.0 - mu
    u_eff = 0.5 * (x * x + yy * yy) + one_minus_mu / r1 + mu / r2 + 0.5 * mu * one_minus_mu
    v2 = vx * vx + vy * vy + vz * vz
    return float(2.0 * u_eff - v2)


# --------------------------------------------------------------------------- #
#  Lagrange points
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LagrangePoints:
    """Locations of the five Lagrange points in synodic non-dim coordinates."""

    L1: tuple[float, float, float]
    L2: tuple[float, float, float]
    L3: tuple[float, float, float]
    L4: tuple[float, float, float]
    L5: tuple[float, float, float]

    def as_array(self) -> np.ndarray:
        """Return the 5 points as a (5, 3) numpy array."""
        return np.array([self.L1, self.L2, self.L3, self.L4, self.L5])


def _collinear_residual(x: float, mu: float) -> float:
    """Quintic in x giving zeros of ∂U/∂x along the x-axis (y = z = 0).

    Roots:  L1 ∈ (−μ, 1−μ),  L2 ∈ (1−μ, ∞),  L3 ∈ (−∞, −μ).
    """
    one_minus_mu = 1.0 - mu
    r1 = abs(x + mu)
    r2 = abs(x - one_minus_mu)
    return x - one_minus_mu * (x + mu) / r1**3 - mu * (x - one_minus_mu) / r2**3


def lagrange_points(mu: float) -> LagrangePoints:
    """Solve for L1..L5 of the CR3BP with mass ratio μ ∈ (0, 0.5].

    L1 / L2 / L3 are roots of a quintic on the x-axis, located on opposite
    sides of the two primaries; L4 / L5 are at the equilateral-triangle
    vertices and have closed-form coordinates.
    """
    if not (0.0 < mu < 0.5):
        raise ValueError(f"mu must lie in (0, 0.5), got {mu}")

    one_minus_mu = 1.0 - mu
    # `brentq` needs a sign change inside the bracket.  We narrow the open
    # intervals away from the singularities at x = −μ and x = 1−μ.
    eps = 1e-9
    x_l1 = brentq(_collinear_residual, -mu + eps, one_minus_mu - eps, args=(mu,))
    x_l2 = brentq(_collinear_residual, one_minus_mu + eps, one_minus_mu + 2.0, args=(mu,))
    x_l3 = brentq(_collinear_residual, -2.0, -mu - eps, args=(mu,))
    # L4 (leading) and L5 (trailing).  Both at distance 1 from each primary,
    # so x = ½ − μ, y = ±√3/2.
    sqrt3_2 = np.sqrt(3.0) / 2.0
    return LagrangePoints(
        L1=(float(x_l1), 0.0, 0.0),
        L2=(float(x_l2), 0.0, 0.0),
        L3=(float(x_l3), 0.0, 0.0),
        L4=(0.5 - mu, sqrt3_2, 0.0),
        L5=(0.5 - mu, -sqrt3_2, 0.0),
    )


# --------------------------------------------------------------------------- #
#  Propagator
# --------------------------------------------------------------------------- #


def propagate_cr3bp(
    state0: np.ndarray,
    t_span: tuple[float, float],
    mu: float,
    steps: int = 400,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a CR3BP trajectory with DOP853.

    Returns ``(t, states_Nx6)`` in synodic non-dimensional coordinates.  Tight
    default tolerances are chosen so the Jacobi constant drifts by <1e-10 over
    one orbital period of an L1 Lyapunov orbit — the integration error budget
    for periodic-orbit work.
    """
    if state0.shape != (6,):
        raise ValueError(f"state0 must have shape (6,); got {state0.shape}")
    sol = solve_ivp(
        cr3bp_rhs,
        t_span,
        state0,
        t_eval=np.linspace(t_span[0], t_span[1], steps),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        args=(mu,),
    )
    if not sol.success:
        raise RuntimeError(f"CR3BP integration failed: {sol.message}")
    return np.asarray(sol.t), np.asarray(sol.y).T


# --------------------------------------------------------------------------- #
#  Dimensionalisation helpers
# --------------------------------------------------------------------------- #


def nondim_to_dim_state(
    state_nd: np.ndarray,
    length_m: float = EM_LENGTH_M,
    mean_motion_rad_s: float = EM_MEAN_MOTION_RAD_S,
) -> np.ndarray:
    """Convert a CR3BP non-dim state to SI synodic coordinates.

    Position scaled by `length_m`; velocity scaled by `length_m · n`.
    """
    out = np.asarray(state_nd, dtype=float).copy()
    out[..., :3] *= length_m
    out[..., 3:] *= length_m * mean_motion_rad_s
    return out


def dim_to_nondim_state(
    state_si: np.ndarray,
    length_m: float = EM_LENGTH_M,
    mean_motion_rad_s: float = EM_MEAN_MOTION_RAD_S,
) -> np.ndarray:
    """Inverse of `nondim_to_dim_state`."""
    out = np.asarray(state_si, dtype=float).copy()
    out[..., :3] /= length_m
    out[..., 3:] /= length_m * mean_motion_rad_s
    return out


# --------------------------------------------------------------------------- #
#  Variational equations (state transition matrix)
# --------------------------------------------------------------------------- #


def cr3bp_jacobian(state: np.ndarray, mu: float) -> np.ndarray:
    """6×6 Jacobian ∂(ẋ)/∂x of the CR3BP RHS at the given state.

    Used by `stm_rhs` to propagate the state transition matrix.  Built from
    the second derivatives of the effective potential U(x,y,z).
    """
    x, y, z, _, _, _ = state
    one_minus_mu = 1.0 - mu
    dx1 = x + mu
    dx2 = x - one_minus_mu
    r1_2 = dx1 * dx1 + y * y + z * z
    r2_2 = dx2 * dx2 + y * y + z * z
    r1_3 = r1_2 ** 1.5
    r2_3 = r2_2 ** 1.5
    r1_5 = r1_2 ** 2.5
    r2_5 = r2_2 ** 2.5

    Uxx = 1.0 - one_minus_mu / r1_3 - mu / r2_3 \
        + 3.0 * one_minus_mu * dx1 * dx1 / r1_5 + 3.0 * mu * dx2 * dx2 / r2_5
    Uyy = 1.0 - one_minus_mu / r1_3 - mu / r2_3 \
        + 3.0 * one_minus_mu * y * y / r1_5 + 3.0 * mu * y * y / r2_5
    Uzz = -one_minus_mu / r1_3 - mu / r2_3 \
        + 3.0 * one_minus_mu * z * z / r1_5 + 3.0 * mu * z * z / r2_5
    Uxy = 3.0 * one_minus_mu * dx1 * y / r1_5 + 3.0 * mu * dx2 * y / r2_5
    Uxz = 3.0 * one_minus_mu * dx1 * z / r1_5 + 3.0 * mu * dx2 * z / r2_5
    Uyz = 3.0 * one_minus_mu * y * z / r1_5 + 3.0 * mu * y * z / r2_5

    A = np.zeros((6, 6))
    A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    A[3, 0], A[3, 1], A[3, 2], A[3, 4] = Uxx, Uxy, Uxz, 2.0
    A[4, 0], A[4, 1], A[4, 2], A[4, 3] = Uxy, Uyy, Uyz, -2.0
    A[5, 0], A[5, 1], A[5, 2] = Uxz, Uyz, Uzz
    return A


def stm_rhs(t: float, y: np.ndarray, mu: float) -> np.ndarray:
    """Combined state + STM RHS for variational integration.

    State vector has 42 components: `y[0:6]` is the spacecraft state and
    `y[6:42]` is the flattened (row-major) 6×6 state transition matrix Φ(t).
    Returns the corresponding derivatives.
    """
    state = y[:6]
    Phi = y[6:].reshape(6, 6)
    state_dot = cr3bp_rhs(t, state, mu)
    A = cr3bp_jacobian(state, mu)
    Phi_dot = A @ Phi
    return np.concatenate([state_dot, Phi_dot.flatten()])


def monodromy_matrix(
    state0: np.ndarray,
    period: float,
    mu: float,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> np.ndarray:
    """Integrate the variational equations around a closed orbit and return
    the monodromy matrix Φ(T)."""
    y0 = np.concatenate([np.asarray(state0, dtype=float), np.eye(6).flatten()])
    sol = solve_ivp(
        stm_rhs, (0.0, period), y0, method="DOP853",
        rtol=rtol, atol=atol, args=(mu,),
    )
    if not sol.success:
        raise RuntimeError(f"monodromy integration failed: {sol.message}")
    return sol.y[6:, -1].reshape(6, 6)


# --------------------------------------------------------------------------- #
#  Periodic-orbit differential correction (planar Lyapunov)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PeriodicOrbit:
    """A closed periodic orbit in the CR3BP synodic frame."""

    state0: tuple[float, float, float, float, float, float]  # IC (non-dim)
    period: float          # full period (non-dim, time units = 1/mean motion)
    jacobi: float          # Jacobi constant
    family: str            # "lyapunov_L1", "lyapunov_L2", ...
    dc_iterations: int     # iterations to converge
    dc_residual: float     # final |vx| at half-period crossing


def find_planar_lyapunov(
    L_point: int,
    Ax: float,
    mu: float = EM_MU,
    max_iter: int = 30,
    tol: float = 1e-11,
) -> PeriodicOrbit:
    """Compute a planar Lyapunov orbit around L1 or L2 via differential
    correction.

    The orbit is parameterised by its x-amplitude `Ax` (non-dim, ≪ 1).  The
    IC has form ``[x0, 0, 0, 0, vy0, 0]`` and the correction iterates vy0
    until the next y=0 crossing has vx = 0 (the symmetry condition that
    closes the orbit).  Uses the analytical STM partial (Howell 1984) so
    convergence is quadratic.
    """
    if L_point not in (1, 2):
        raise ValueError(f"L_point must be 1 or 2, got {L_point}")
    if Ax <= 0:
        raise ValueError(f"Ax must be > 0, got {Ax}")

    L = lagrange_points(mu)
    lx = L.L1[0] if L_point == 1 else L.L2[0]

    # Initial guess: place the spacecraft at the outer extreme of the orbit
    # and give it a transverse kick consistent with the linearised Lyapunov
    # motion.  L1 and L2 orbits have opposite chirality (Coriolis-induced):
    #   L1: x0 = L1 − Ax (toward Earth), vy0 > 0 (counter-clockwise from above)
    #   L2: x0 = L2 + Ax (away from Moon), vy0 < 0 (clockwise from above)
    # |vy0|/Ax ≈ κ·ω_p ≈ 6 from linear theory at both collinear points.
    x0 = lx - Ax if L_point == 1 else lx + Ax
    seed_vy = +Ax * 6.0 if L_point == 1 else -Ax * 6.0
    state = np.array([x0, 0.0, 0.0, 0.0, seed_vy, 0.0])

    # Integration horizon: must exceed the half-period.  EM L1/L2 Lyapunov
    # orbits have non-dim periods around 2.7--3.4, so 6.0 is safe.
    T_HORIZON = 6.0
    T_SEARCH_MIN = 0.1
    iteration = 0
    residual = float("inf")
    for iteration in range(max_iter):
        y0 = np.concatenate([state, np.eye(6).flatten()])
        sol = solve_ivp(
            stm_rhs, (0.0, T_HORIZON), y0, method="DOP853",
            rtol=1e-12, atol=1e-12, args=(mu,), dense_output=True,
        )
        if not sol.success:
            raise RuntimeError(f"Lyapunov DC: integration failed ({sol.message})")

        # Find the first descending y=0 crossing past T_SEARCH_MIN by scanning
        # the dense output for a sign change, then root-finding inside it.
        sol_loop = sol  # explicit capture to satisfy ruff B023
        def _y_of(t: float, _s=sol_loop) -> float:
            return float(_s.sol(t)[1])

        scan_t = np.linspace(T_SEARCH_MIN, T_HORIZON, 400)
        scan_y = np.array([_y_of(t) for t in scan_t])
        # Any sign change marks the next y=0 crossing — that's the half-period
        # regardless of whether the orbit was above or below the x-axis.
        sign_change = np.where(scan_y[:-1] * scan_y[1:] < 0)[0]
        if not sign_change.size:
            raise RuntimeError("Lyapunov DC: no y=0 crossing in horizon")
        i0 = int(sign_change[0])
        T_half = float(brentq(_y_of, scan_t[i0], scan_t[i0 + 1], xtol=1e-13))
        y_at_half = sol.sol(T_half)
        state_half = y_at_half[:6]
        Phi = y_at_half[6:].reshape(6, 6)

        vx_half = float(state_half[3])
        residual = abs(vx_half)
        if residual < tol:
            period = 2.0 * T_half
            return PeriodicOrbit(
                state0=tuple(state.tolist()),  # type: ignore[arg-type]
                period=period,
                jacobi=jacobi_constant(state, mu),
                family=f"lyapunov_L{L_point}",
                dc_iterations=iteration,
                dc_residual=residual,
            )

        # Newton step in vy0 (Howell 1984):
        #     δvy0 = -vx_half / (Phi[3,4] − Phi[1,4] · ax_half / vy_half)
        rhs_full = cr3bp_rhs(T_half, state_half, mu)
        ax_half = float(rhs_full[3])
        vy_half = float(state_half[4])
        if abs(vy_half) < 1e-12:
            raise RuntimeError("Lyapunov DC: vy_half too small for Newton step")
        denom = Phi[3, 4] - Phi[1, 4] * ax_half / vy_half
        if abs(denom) < 1e-14:
            raise RuntimeError("Lyapunov DC: singular Newton denominator")
        state[4] += -vx_half / denom

    raise RuntimeError(
        f"Lyapunov DC did not converge in {max_iter} iter; residual={residual:.3e}"
    )


# --------------------------------------------------------------------------- #
#  Invariant manifolds
# --------------------------------------------------------------------------- #


def manifold_eigendirections(
    monodromy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Extract the stable and unstable eigenvectors of a monodromy matrix.

    Returns ``(v_unstable, v_stable, λ_unstable, λ_stable)`` with each
    eigenvector real-valued and unit-norm.  For a hyperbolic CR3BP periodic
    orbit the monodromy spectrum factors as (λ, 1/λ, 1, 1, e^±iθ); we sort by
    |λ| and pick the extremes, then verify they're approximately real
    (any tiny imaginary part is numerical noise from `np.linalg.eig`).
    """
    eigvals, eigvecs = np.linalg.eig(monodromy)
    order = np.argsort(-np.abs(eigvals))  # descending |λ|
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    def _real_or_die(lam: complex, vec: np.ndarray, tag: str) -> tuple[float, np.ndarray]:
        if abs(lam.imag) > 1e-3 * max(abs(lam.real), 1.0):
            raise RuntimeError(f"{tag} eigenvalue is complex (λ={lam}); orbit may be elliptic")
        v = vec.real
        n = np.linalg.norm(v)
        if n < 1e-12:
            raise RuntimeError(f"{tag} eigenvector has near-zero real part")
        return float(lam.real), v / n

    lam_u, v_u = _real_or_die(eigvals[0], eigvecs[:, 0], "unstable")
    lam_s, v_s = _real_or_die(eigvals[-1], eigvecs[:, -1], "stable")
    return v_u, v_s, lam_u, lam_s


def compute_manifold(
    state0: np.ndarray,
    period: float,
    mu: float,
    direction: str = "unstable",
    branch: str = "+",
    n_samples: int = 40,
    duration: float = 8.0,
    perturbation: float = 1e-6,
    steps: int = 400,
) -> list[np.ndarray]:
    """Integrate one branch of an invariant manifold of a periodic orbit.

    Workflow:
      1. Sample `n_samples` points evenly distributed along the orbit
      2. At each, perturb the state along the chosen eigenvector
      3. Integrate forward (unstable) or backward (stable) for `duration`
    Returns a list of (steps, 6) state arrays, one per sampled point.
    """
    if direction not in ("stable", "unstable"):
        raise ValueError("direction must be 'stable' or 'unstable'")
    if branch not in ("+", "-"):
        raise ValueError("branch must be '+' or '-'")
    sign = 1.0 if branch == "+" else -1.0

    # 1. Sample N points around the orbit and the STM at each.
    y0 = np.concatenate([np.asarray(state0, dtype=float), np.eye(6).flatten()])
    sample_times = np.linspace(0.0, period, n_samples, endpoint=False)
    sol = solve_ivp(
        stm_rhs, (0.0, period), y0, t_eval=sample_times, method="DOP853",
        rtol=1e-12, atol=1e-12, args=(mu,),
    )
    if not sol.success:
        raise RuntimeError(f"orbit sampling failed: {sol.message}")

    # 2. Eigenvectors at t=0; transport to each sample via the STM.
    monodromy = monodromy_matrix(np.asarray(state0, dtype=float), period, mu)
    v_u, v_s, _, _ = manifold_eigendirections(monodromy)
    v0 = v_u if direction == "unstable" else v_s

    # 3. Integrate each branch.  Unstable → forward in time; stable → backward.
    out: list[np.ndarray] = []
    t_int = duration if direction == "unstable" else -duration
    eval_pts = np.linspace(0.0, t_int, steps)
    for i in range(n_samples):
        state_i = sol.y[:6, i]
        Phi_i = sol.y[6:, i].reshape(6, 6)
        v_i = Phi_i @ v0
        v_i = v_i / np.linalg.norm(v_i)
        seed = state_i + sign * perturbation * v_i
        branch_sol = solve_ivp(
            cr3bp_rhs, (0.0, t_int), seed, t_eval=eval_pts,
            method="DOP853", rtol=1e-10, atol=1e-12, args=(mu,),
        )
        if not branch_sol.success:
            # Skip trajectories that hit a singularity instead of failing the
            # whole batch — close passes to the primaries can blow up.
            continue
        out.append(branch_sol.y.T)
    return out


# --------------------------------------------------------------------------- #
#  Weak Stability Boundary — coarse capture/escape diagnostic (Phase 4.8)
# --------------------------------------------------------------------------- #


def wsb_capture_grid(
    altitudes_m: np.ndarray,
    angles_rad: np.ndarray,
    mu: float = EM_MU,
    length_m: float = EM_LENGTH_M,
    moon_radius_m: float = 1_737_400.0,
    duration: float = 6.0,
    escape_radius: float = 2.0,
) -> np.ndarray:
    """Crude WSB diagnostic in the CR3BP.

    For each (altitude, angle) pair, place a test particle on a circular
    velocity prograde around the Moon, propagate backward in time, and
    classify as "captured" (stayed within `escape_radius` of the system
    barycenter for the entire integration) or "escaped".

    Returns a (len(altitudes), len(angles)) integer grid:
      1 = captured-from-past (interesting WSB trajectory)
      0 = escaped (came in from infinity)
     −1 = blew up / integrator failed (treat as captured for plotting)
    """
    out = np.zeros((len(altitudes_m), len(angles_rad)), dtype=int)
    moon_x = 1.0 - mu
    for i, alt_m in enumerate(altitudes_m):
        # Non-dim radius around the Moon.
        r_nd = (moon_radius_m + alt_m) / length_m
        # Circular velocity around the Moon (Keplerian, in Moon-centric frame),
        # converted to non-dim units.  Synodic frame correction: subtract the
        # rotating-frame velocity ω × r = (1)·r_nd along ŷ.
        v_kepler = np.sqrt(mu / r_nd)
        for j, theta in enumerate(angles_rad):
            x = moon_x + r_nd * np.cos(theta)
            y = r_nd * np.sin(theta)
            # Prograde tangential velocity in inertial → subtract synodic rotation.
            vx_inert = -v_kepler * np.sin(theta)
            vy_inert = v_kepler * np.cos(theta)
            # Synodic-frame velocity = inertial − ω × r (ω=1 about +z).
            vx = vx_inert - (-y)
            vy = vy_inert - x
            state0 = np.array([x, y, 0.0, vx, vy, 0.0])
            try:
                # Backward propagation.
                sol = solve_ivp(
                    cr3bp_rhs, (0.0, -duration), state0, method="DOP853",
                    rtol=1e-9, atol=1e-10, args=(mu,),
                )
                if not sol.success:
                    out[i, j] = -1
                    continue
                radii = np.linalg.norm(sol.y[:3, :], axis=0)
                out[i, j] = 1 if radii.max() < escape_radius else 0
            except Exception:
                out[i, j] = -1
    return out
