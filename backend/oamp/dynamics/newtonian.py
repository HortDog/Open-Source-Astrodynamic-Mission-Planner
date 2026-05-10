"""Newtonian two-body propagator (placeholder MVP integrator).

Integration parameter is coordinate time (treat as TDB at this stage); state
is BCRS/GCRS-agnostic and assumed consistent with the chosen mu.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy.integrate import solve_ivp


class TwoBodyState(BaseModel):
    r: tuple[float, float, float]  # m
    v: tuple[float, float, float]  # m/s


def _rhs(_t: float, y: np.ndarray, mu: float) -> np.ndarray:
    r = y[:3]
    v = y[3:]
    a = -mu * r / np.linalg.norm(r) ** 3
    return np.concatenate((v, a))


def propagate_two_body(
    state: TwoBodyState,
    duration_s: float,
    steps: int = 200,
    mu: float = 3.986004418e14,
) -> tuple[np.ndarray, np.ndarray]:
    y0 = np.array([*state.r, *state.v], dtype=float)
    t_eval = np.linspace(0.0, duration_s, steps)
    sol = solve_ivp(
        _rhs,
        (0.0, duration_s),
        y0,
        t_eval=t_eval,
        args=(mu,),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    return sol.t, sol.y.T
