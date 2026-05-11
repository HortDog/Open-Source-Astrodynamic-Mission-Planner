"""Newtonian orbital propagator.

The propagator integrates two-body gravity plus an arbitrary list of
perturbations from `oamp.dynamics.perturbations`. Impulsive Δv manoeuvres
are applied between integration arcs so the ODE itself stays smooth.

Coordinate frame: body-centred inertial whose Z axis is the body's rotation
pole (GCRS/J2000 for Earth). Integration parameter is TDB seconds since
J2000, matching SPICE ET — every perturbation receives the absolute epoch
plus a relative time offset built in by the caller (see `propagate_orbit`).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from scipy.integrate import solve_ivp

from oamp.bodies import EARTH, Body
from oamp.dynamics.perturbations import Perturbation, zonal_harmonics


class TwoBodyState(BaseModel):
    r: tuple[float, float, float]  # m
    v: tuple[float, float, float]  # m/s


class Maneuver(BaseModel):
    """Instantaneous Δv applied at `t_offset_s` seconds after t0.

    Expressed in the RIC (radial / in-track / cross-track) local frame so that
    the same descriptor stays meaningful regardless of inertial frame choice.
    """

    t_offset_s: float = Field(ge=0.0)
    dv_ric: tuple[float, float, float]  # m/s


def two_body_acceleration(r: np.ndarray, mu: float) -> np.ndarray:
    return -mu * r / np.linalg.norm(r) ** 3


def j2_acceleration(r: np.ndarray, mu: float, body_radius: float, j2: float) -> np.ndarray:
    """Back-compat helper kept for the existing tests; equivalent to the J2
    term inside `oamp.dynamics.perturbations._accel_j2`."""
    if j2 == 0.0:
        return np.zeros(3)
    x, y, z = r
    rn = float(np.linalg.norm(r))
    k = -1.5 * j2 * mu * body_radius**2 / rn**5
    z2r2 = (z * z) / (rn * rn)
    return np.array(
        [
            k * (1 - 5 * z2r2) * x,
            k * (1 - 5 * z2r2) * y,
            k * (3 - 5 * z2r2) * z,
        ]
    )


# --------------------------------------------------------------------------- #
#  Frame helpers
# --------------------------------------------------------------------------- #


def ric_to_inertial(r: np.ndarray, v: np.ndarray, dv_ric: np.ndarray) -> np.ndarray:
    """Convert a Δv expressed in the RIC frame to inertial XYZ.

    R̂  = r / |r|        (radial)
    Ĉ  = (r × v)/|...|   (cross-track / orbit normal)
    Î  = Ĉ × R̂          (in-track, completes right-handed triad)
    """
    r_hat = r / np.linalg.norm(r)
    c_hat = np.cross(r, v)
    c_hat = c_hat / np.linalg.norm(c_hat)
    i_hat = np.cross(c_hat, r_hat)
    R = np.column_stack((r_hat, i_hat, c_hat))  # columns are basis vectors
    return R @ dv_ric


# --------------------------------------------------------------------------- #
#  Core propagator
# --------------------------------------------------------------------------- #


def _make_rhs(
    mu: float,
    perturbation: Perturbation | None,
    t0: float,
):
    """Return the SciPy-style RHS closure.

    `t0` is the absolute TDB epoch corresponding to integration time 0; the
    perturbation callable receives the absolute epoch so SPICE-based
    perturbations (3rd-body, SRP) can query ephemerides correctly.
    """
    if perturbation is None:

        def _rhs(_t: float, y: np.ndarray) -> np.ndarray:
            r, v = y[:3], y[3:]
            return np.concatenate((v, two_body_acceleration(r, mu)))
    else:

        def _rhs(t: float, y: np.ndarray) -> np.ndarray:
            r, v = y[:3], y[3:]
            a = two_body_acceleration(r, mu) + perturbation(t0 + t, r, v)
            return np.concatenate((v, a))

    return _rhs


def propagate_orbit(
    state: TwoBodyState,
    duration_s: float,
    steps: int = 200,
    body: Body = EARTH,
    j2_enabled: bool = False,
    perturbation: Perturbation | None = None,
    maneuvers: list[Maneuver] | None = None,
    t0_tdb: float = 0.0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate `state` over `duration_s` and return (times, states_Nx6).

    Parameters
    ----------
    state
        Initial inertial state in m / m/s.
    duration_s
        Total propagation span in seconds.
    steps
        Number of evaluation points returned (uniformly sampled).
    body
        Central body whose μ, equatorial radius, and zonal coefficients are used.
    j2_enabled
        Convenience flag — when true and no explicit `perturbation` is supplied,
        a J2-only perturbation is built from the body's `J2` constant.
    perturbation
        Optional callable (t, r, v) → a (m/s²). If provided it supersedes the
        `j2_enabled` shortcut and is the only perturbation applied.
    maneuvers
        Sorted list of impulsive Δv kicks. Each manoeuvre splits the integration
        into a new arc so the ODE itself remains smooth.
    t0_tdb
        Absolute TDB epoch for `t = 0`. Forwarded to perturbations that need
        ephemerides (3rd-body gravity, SRP).
    rtol, atol
        Relative / absolute tolerance for DOP853.
    """
    # Build the effective perturbation if the caller asked for the J2 shortcut.
    if perturbation is None and j2_enabled and body.j2 != 0.0:
        perturbation = zonal_harmonics(body, n_max=2)

    mans = sorted(maneuvers or [], key=lambda m: m.t_offset_s)
    if any(m.t_offset_s > duration_s for m in mans):
        raise ValueError("manoeuvre occurs after the requested propagation end")

    # Split the span at manoeuvre epochs so the ODE stays smooth across each arc.
    breakpoints = [0.0, *(m.t_offset_s for m in mans), duration_s]

    # Distribute output samples across arcs in proportion to their duration.
    total = duration_s if duration_s > 0 else 1.0
    arc_samples = [
        max(2, int(round(steps * (breakpoints[i + 1] - breakpoints[i]) / total)))
        for i in range(len(breakpoints) - 1)
    ]

    times_chunks: list[np.ndarray] = []
    state_chunks: list[np.ndarray] = []
    y = np.array([*state.r, *state.v], dtype=float)
    rhs = _make_rhs(body.mu, perturbation, t0_tdb)

    for i, n_samples in enumerate(arc_samples):
        t_a, t_b = breakpoints[i], breakpoints[i + 1]
        if t_b <= t_a:
            continue
        sol = solve_ivp(
            rhs,
            (t_a, t_b),
            y,
            t_eval=np.linspace(t_a, t_b, n_samples),
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        # Drop the duplicate sample at every internal join.
        if i == 0:
            times_chunks.append(sol.t)
            state_chunks.append(sol.y.T)
        else:
            times_chunks.append(sol.t[1:])
            state_chunks.append(sol.y.T[1:])

        # Apply the impulsive manoeuvre at the right-hand boundary, if any.
        y = sol.y[:, -1].copy()
        if i < len(mans):
            dv_inertial = ric_to_inertial(y[:3], y[3:], np.asarray(mans[i].dv_ric))
            y[3:] = y[3:] + dv_inertial

    return np.concatenate(times_chunks), np.vstack(state_chunks)


# --------------------------------------------------------------------------- #
#  Back-compat shim used by existing tests / old API
# --------------------------------------------------------------------------- #


def propagate_two_body(
    state: TwoBodyState,
    duration_s: float,
    steps: int = 200,
    mu: float = EARTH.mu,
) -> tuple[np.ndarray, np.ndarray]:
    body = Body(name="custom", mu=mu, radius=EARTH.radius, j2=0.0)
    return propagate_orbit(state, duration_s, steps, body=body, j2_enabled=False)
