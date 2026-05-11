"""Fuel-optimal trajectory NLP via CasADi.

Multi-shooting formulation: given an initial state x0, a target state x_f, and
a list of N manoeuvre epochs t_1..t_N, find the impulsive Δv vectors that
minimise total fuel ‖Σ|Δv_i|‖ subject to two-body dynamics linking the burns.

Two-body propagation between burns is built as a CasADi `integrator('cvodes')`
so the whole problem stays differentiable and IPOPT can take Newton steps on
the optimality conditions.

Higher-fidelity dynamics (J2, drag, SRP) can be added by composing CasADi
expressions for those forces, but for Phase 2 the two-body propagator is the
practical workhorse — it agrees with the full perturbed dynamics to ≲ km over
typical transfer durations and converges quickly from a Lambert warm-start.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from oamp.bodies import EARTH

Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    dv_inertial_m_s: list[Vec3]  # Δv at each manoeuvre epoch, inertial frame
    total_dv_m_s: float
    converged: bool
    iterations: int
    final_state_m: Vec3  # propagated terminal r (sanity-check)
    final_velocity_m_s: Vec3  # propagated terminal v


def _two_body_dae(mu: float) -> dict:
    """Build the CasADi DAE dict for two-body dynamics."""
    x = ca.MX.sym("x", 6)
    r = x[0:3]
    v = x[3:6]
    r_norm = ca.norm_2(r)
    a = -mu * r / r_norm**3
    rhs = ca.vertcat(v, a)
    return {"x": x, "ode": rhs}


def optimize_multi_burn(
    x0_r: Vec3,
    x0_v: Vec3,
    xf_r: Vec3,
    xf_v: Vec3,
    maneuver_epochs_s: list[float],
    t_final_s: float,
    mu: float = EARTH.mu,
    initial_dv_guess: list[Vec3] | None = None,
    print_level: int = 0,
) -> OptimizationResult:
    """Minimise total Δv across a fixed-epoch impulsive trajectory.

    Parameters
    ----------
    x0_r, x0_v
        Initial inertial position (m) and velocity (m/s).
    xf_r, xf_v
        Required terminal inertial position (m) and velocity (m/s).
    maneuver_epochs_s
        Strictly increasing list of manoeuvre times (seconds from t=0). Must
        all be strictly between 0 and `t_final_s`.
    t_final_s
        Time horizon at which the propagated state must match (xf_r, xf_v).
    mu
        Gravitational parameter, m³/s².
    initial_dv_guess
        Optional list of N Cartesian Δv guesses (m/s) used to warm-start
        IPOPT. Pass the Lambert-derived guess for best convergence.
    print_level
        IPOPT verbosity (0 = silent, 5 = verbose).

    Notes
    -----
    With N burns the NLP has 3N decision variables and 6 equality constraints
    (terminal state match). For N=2 the problem is determined (no optimisation
    freedom — same as Lambert); for N≥3 IPOPT minimises ‖Δv‖.
    """
    n = len(maneuver_epochs_s)
    if n < 1:
        raise ValueError("need at least one manoeuvre epoch")
    eps_sorted = sorted(maneuver_epochs_s)
    if eps_sorted != list(maneuver_epochs_s):
        raise ValueError("manoeuvre epochs must be strictly increasing")
    if eps_sorted[0] <= 0.0 or eps_sorted[-1] > t_final_s:
        raise ValueError("manoeuvre epochs must lie in (0, t_final_s] (last may equal t_final)")

    # Build a CasADi integrator for each segment between consecutive epochs.
    # We use the explicit fixed-step Runge–Kutta integrator: cvodes' adaptive
    # step control was unreliable inside IPOPT's line search (it triggered
    # CV_TOO_MUCH_WORK for trial Δv values that briefly send the spacecraft
    # near escape). RK4 with a fixed step count is deterministic and fast for
    # the transfer-length arcs typical of Phase 2 problems.
    boundaries = [0.0, *maneuver_epochs_s, t_final_s]
    # Drop any zero-duration trailing segment (when the last manoeuvre lands at
    # t_final the post-burn coast is degenerate and CasADi rejects 0-length).
    boundaries = [t for i, t in enumerate(boundaries) if i == 0 or t > boundaries[i - 1] + 1e-9]
    dae = _two_body_dae(mu)
    segments = []
    for i in range(len(boundaries) - 1):
        ti, tj = boundaries[i], boundaries[i + 1]
        # ~100 RK4 sub-steps per orbit → < 1 m error for transfer arcs.
        substeps = max(50, int((tj - ti) / 60.0))
        F = ca.integrator(
            f"F_{i}",
            "rk",
            dae,
            ti,
            tj,
            {"number_of_finite_elements": substeps},
        )
        segments.append(F)

    # NLP via Opti stack.
    opti = ca.Opti()
    DV = opti.variable(3, n)  # decision variables: Δv vectors

    # Bound each component to ±15 km/s — physically unreachable per single
    # impulse for any chemical propulsion, and keeps IPOPT trial steps inside
    # the integrator's stable region.
    opti.subject_to(opti.bounded(-15_000.0, DV, 15_000.0))

    # Forward-shoot the trajectory with symbolic manoeuvres.
    x = ca.DM([*x0_r, *x0_v])
    for i in range(n):
        res = segments[i](x0=x)
        x_pre = res["xf"]
        # Apply Δv (additive on velocity).
        x = ca.vertcat(x_pre[0:3], x_pre[3:6] + DV[:, i])
    # Tail coast: if there is a segment past the last manoeuvre, propagate it.
    if len(segments) > n:
        x = segments[n](x0=x)["xf"]
    x_f_sym = x

    # Equality constraint: terminal state match.
    target = ca.DM([*xf_r, *xf_v])
    opti.subject_to(x_f_sym == target)

    # Cost: total Δv magnitude. Use a smoothed norm to keep gradients well-defined
    # near zero (sqrt(ε² + |Δv|²) — converges to |Δv| as ε→0).
    eps = 1e-6
    fuel = 0
    for i in range(n):
        fuel = fuel + ca.sqrt(eps**2 + DV[0, i] ** 2 + DV[1, i] ** 2 + DV[2, i] ** 2)
    opti.minimize(fuel)

    # Initial guess.
    if initial_dv_guess is not None:
        if len(initial_dv_guess) != n:
            raise ValueError(f"initial_dv_guess length {len(initial_dv_guess)} != n {n}")
        guess = np.asarray(initial_dv_guess, dtype=float).T  # shape (3, n)
        opti.set_initial(DV, guess)
    else:
        opti.set_initial(DV, np.zeros((3, n)))

    opti.solver(
        "ipopt",
        {"print_time": False},
        {
            "print_level": print_level,
            "max_iter": 200,
            "tol": 1e-8,
            "acceptable_tol": 1e-6,
            # CVODES backwards-mode second-order sensitivities are unreliable for
            # our use case; L-BFGS quasi-Newton avoids them and converges in a
            # comparable number of iterations for this small NLP.
            "hessian_approximation": "limited-memory",
        },
    )

    try:
        sol = opti.solve()
        converged = True
        iterations = int(opti.stats().get("iter_count", 0))
        dv_sol = np.asarray(sol.value(DV)).reshape(3, n)
        # Recompute the terminal state for the response so the caller can verify.
        x_final = np.asarray(sol.value(x_f_sym)).flatten()
    except Exception:
        converged = False
        iterations = int(opti.stats().get("iter_count", 0))
        try:
            dv_sol = np.asarray(opti.debug.value(DV)).reshape(3, n)
            x_final = np.asarray(opti.debug.value(x_f_sym)).flatten()
        except Exception:
            dv_sol = np.zeros((3, n))
            x_final = np.zeros(6)

    dv_list: list[Vec3] = [
        (float(dv_sol[0, i]), float(dv_sol[1, i]), float(dv_sol[2, i])) for i in range(n)
    ]
    total_dv = float(sum(np.linalg.norm(dv_sol[:, i]) for i in range(n)))
    return OptimizationResult(
        dv_inertial_m_s=dv_list,
        total_dv_m_s=total_dv,
        converged=converged,
        iterations=iterations,
        final_state_m=(float(x_final[0]), float(x_final[1]), float(x_final[2])),
        final_velocity_m_s=(float(x_final[3]), float(x_final[4]), float(x_final[5])),
    )
