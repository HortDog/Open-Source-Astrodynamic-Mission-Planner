"""Newtonian orbital propagator with optional J2 oblateness term.

Integration parameter is coordinate time (treat as TDB). State and the J2
axis are assumed to be in an inertial frame whose Z axis is the body's
rotation pole — for Earth that's J2000 / GCRS (close enough for MVP).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy.integrate import solve_ivp

from oamp.bodies import EARTH, Body


class TwoBodyState(BaseModel):
    r: tuple[float, float, float]  # m
    v: tuple[float, float, float]  # m/s


def two_body_acceleration(r: np.ndarray, mu: float) -> np.ndarray:
    return -mu * r / np.linalg.norm(r) ** 3


def j2_acceleration(r: np.ndarray, mu: float, body_radius: float, j2: float) -> np.ndarray:
    """Acceleration due to the J2 zonal harmonic (Earth oblateness term).

    Vallado §2.3, eq. 2-3:
        a = -(3/2) * J2 * mu * R^2 / r^5 * [
                (1 - 5 z^2 / r^2) x,
                (1 - 5 z^2 / r^2) y,
                (3 - 5 z^2 / r^2) z,
            ]
    Returns zero when J2 is zero so the same propagator handles both regimes.
    """
    if j2 == 0.0:
        return np.zeros(3)
    x, y, z = r
    r_norm = float(np.linalg.norm(r))
    factor = -1.5 * j2 * mu * body_radius**2 / r_norm**5
    z2_over_r2 = (z * z) / (r_norm * r_norm)
    return np.array([
        factor * (1 - 5 * z2_over_r2) * x,
        factor * (1 - 5 * z2_over_r2) * y,
        factor * (3 - 5 * z2_over_r2) * z,
    ])


def _rhs(_t: float, y: np.ndarray, mu: float, body_radius: float, j2: float) -> np.ndarray:
    r = y[:3]
    v = y[3:]
    a = two_body_acceleration(r, mu) + j2_acceleration(r, mu, body_radius, j2)
    return np.concatenate((v, a))


def propagate_orbit(
    state: TwoBodyState,
    duration_s: float,
    steps: int = 200,
    body: Body = EARTH,
    j2_enabled: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a state over `duration_s` and return (times, states_Nx6).

    The default uses pure two-body gravity; pass `j2_enabled=True` to include
    the J2 zonal-harmonic perturbation for the chosen body.
    """
    y0 = np.array([*state.r, *state.v], dtype=float)
    t_eval = np.linspace(0.0, duration_s, steps)
    j2 = body.j2 if j2_enabled else 0.0
    sol = solve_ivp(
        _rhs,
        (0.0, duration_s),
        y0,
        t_eval=t_eval,
        args=(body.mu, body.radius, j2),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    return sol.t, sol.y.T


# Back-compat shim: the existing /propagate endpoint and tests still call this.
def propagate_two_body(
    state: TwoBodyState,
    duration_s: float,
    steps: int = 200,
    mu: float = EARTH.mu,
) -> tuple[np.ndarray, np.ndarray]:
    body = Body(name="custom", mu=mu, radius=EARTH.radius, j2=0.0)
    return propagate_orbit(state, duration_s, steps, body=body, j2_enabled=False)
