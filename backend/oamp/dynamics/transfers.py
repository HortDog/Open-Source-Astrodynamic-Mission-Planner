"""Closed-form and boundary-value transfer solvers.

- `hohmann_transfer`  — coplanar circular-to-circular two-impulse, closed form
- `lambert_universal` — Izzo-style universal-variable Lambert solver for the
  two-point boundary-value problem (find v1, v2 given r1, r2, Δt).

The Lambert routine is the practical workhorse for trajectory design — it
gives a fast, robust initial guess that downstream CasADi optimisation can
warm-start from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from oamp.bodies import EARTH

# --------------------------------------------------------------------------- #
#  Hohmann transfer  (analytic, coplanar, circular orbits)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HohmannResult:
    dv1_m_s: float  # impulse at the inner orbit
    dv2_m_s: float  # impulse at the outer orbit
    dv_total_m_s: float
    transfer_time_s: float
    semi_major_axis_m: float


def hohmann_transfer(r1_m: float, r2_m: float, mu: float = EARTH.mu) -> HohmannResult:
    """Two-impulse coplanar transfer between circular orbits of radii r1 and r2."""
    if r1_m <= 0 or r2_m <= 0:
        raise ValueError("radii must be positive")
    a_t = 0.5 * (r1_m + r2_m)
    v1 = math.sqrt(mu / r1_m)
    v2 = math.sqrt(mu / r2_m)
    v_peri = math.sqrt(mu * (2 / r1_m - 1 / a_t))
    v_apo = math.sqrt(mu * (2 / r2_m - 1 / a_t))
    dv1 = abs(v_peri - v1)
    dv2 = abs(v2 - v_apo)
    tof = math.pi * math.sqrt(a_t**3 / mu)
    return HohmannResult(
        dv1_m_s=dv1,
        dv2_m_s=dv2,
        dv_total_m_s=dv1 + dv2,
        transfer_time_s=tof,
        semi_major_axis_m=a_t,
    )


# --------------------------------------------------------------------------- #
#  Lambert  (universal-variable / Battin-Vaughan formulation)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LambertResult:
    v1_m_s: tuple[float, float, float]
    v2_m_s: tuple[float, float, float]
    iterations: int
    converged: bool
    transfer_time_s: float


def _stumpff_c(z: float) -> float:
    if z > 1e-6:
        s = math.sqrt(z)
        return (1 - math.cos(s)) / z
    if z < -1e-6:
        s = math.sqrt(-z)
        return (1 - math.cosh(s)) / z
    return 0.5 - z / 24.0 + z * z / 720.0


def _stumpff_s(z: float) -> float:
    if z > 1e-6:
        s = math.sqrt(z)
        return (s - math.sin(s)) / (s**3)
    if z < -1e-6:
        s = math.sqrt(-z)
        return (math.sinh(s) - s) / (s**3)
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


def lambert_universal(
    r1: tuple[float, float, float] | np.ndarray,
    r2: tuple[float, float, float] | np.ndarray,
    tof_s: float,
    mu: float = EARTH.mu,
    prograde: bool = True,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> LambertResult:
    """Solve Lambert's problem: find v1, v2 connecting r1→r2 in `tof_s`.

    Universal-variable formulation following Curtis 4e §5.3. Handles elliptic,
    parabolic, and hyperbolic arcs uniformly. The 180° transfer (r1·r2 = −|r1|·|r2|)
    is genuinely degenerate (plane of transfer is undefined) and raises.

    `prograde=True` picks the transfer going in the same sense as r1 × r2 z-component;
    set to False to force the retrograde branch.
    """
    r1v = np.asarray(r1, dtype=float)
    r2v = np.asarray(r2, dtype=float)
    r1n = float(np.linalg.norm(r1v))
    r2n = float(np.linalg.norm(r2v))

    cos_dnu = float(np.dot(r1v, r2v) / (r1n * r2n))
    cos_dnu = max(-1.0, min(1.0, cos_dnu))
    if 1 - cos_dnu < 1e-9:
        raise ValueError("colinear (Δν ≈ 0) Lambert geometry")
    if 1 + cos_dnu < 1e-9:
        raise ValueError("180° transfer is degenerate (transfer plane undefined)")

    cross_z = float(np.cross(r1v, r2v)[2])
    base = math.acos(cos_dnu)
    if prograde:
        d_nu = base if cross_z >= 0 else 2 * math.pi - base
    else:
        d_nu = 2 * math.pi - base if cross_z >= 0 else base
    sin_dnu = math.sin(d_nu)

    # Curtis 5.38: A = sin Δν * sqrt(r1 r2 / (1 - cos Δν)). Sign on A absorbs
    # the prograde/retrograde choice via sin_dnu.
    A = sin_dnu * math.sqrt(r1n * r2n / (1 - cos_dnu))
    if A == 0.0:
        raise ValueError("degenerate Lambert geometry (A = 0)")

    # Define F(z) = √μ · tof(z) − √μ · tof_target, where tof(z) is the time
    # corresponding to universal anomaly z. We solve F(z) = 0 by bracketed
    # Brent root-finding which is robust across elliptic/parabolic/hyperbolic
    # branches without needing a tuned initial guess.
    sqrt_mu_tof = math.sqrt(mu) * tof_s

    def F(z: float) -> float:
        C = _stumpff_c(z)
        S = _stumpff_s(z)
        if C <= 0:
            return math.inf
        y = r1n + r2n + A * (z * S - 1) / math.sqrt(C)
        if y <= 0:
            return -math.inf
        x = math.sqrt(y / C)
        return x**3 * S + A * math.sqrt(y) - sqrt_mu_tof

    # Bracket z. Bound: −4π² (deep hyperbolic) to 4π² (one revolution).
    z_lo, z_hi = -4 * math.pi * math.pi, 4 * math.pi * math.pi - 1e-3
    f_lo, f_hi = F(z_lo), F(z_hi)
    # Walk inward if either endpoint is degenerate (inf / nan).
    while not math.isfinite(f_lo):
        z_lo += 1.0
        f_lo = F(z_lo)
        if z_lo >= z_hi:
            raise RuntimeError("Lambert solver could not bracket z (low side)")
    while not math.isfinite(f_hi):
        z_hi -= 0.5
        f_hi = F(z_hi)
        if z_hi <= z_lo:
            raise RuntimeError("Lambert solver could not bracket z (high side)")
    if f_lo * f_hi > 0:
        raise RuntimeError(
            f"Lambert solver: requested tof ({tof_s:.1f}s) is outside the "
            "feasible range for the chosen transfer branch"
        )

    try:
        z = brentq(F, z_lo, z_hi, xtol=1e-12, maxiter=max_iter)
        converged = True
        it = max_iter  # brentq does not expose iteration count
    except Exception as e:
        raise RuntimeError(f"Lambert root-finding failed: {e}") from e

    C = _stumpff_c(z)
    S = _stumpff_s(z)
    y = r1n + r2n + A * (z * S - 1) / math.sqrt(C)

    f = 1 - y / r1n
    g = A * math.sqrt(y / mu)
    g_dot = 1 - y / r2n
    if g == 0.0:
        raise RuntimeError("Lambert solver hit degenerate g coefficient")
    v1 = (r2v - f * r1v) / g
    v2 = (g_dot * r2v - r1v) / g

    return LambertResult(
        v1_m_s=(float(v1[0]), float(v1[1]), float(v1[2])),
        v2_m_s=(float(v2[0]), float(v2[1]), float(v2[2])),
        iterations=it,
        converged=converged,
        transfer_time_s=tof_s,
    )
