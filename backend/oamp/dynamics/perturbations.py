"""Orbital perturbation accelerations.

Each perturbation is a Protocol-conforming callable taking (t, r, v, ctx) and
returning an acceleration vector in m/s². The propagator sums them on top of
the two-body central force. This keeps the perturbation set composable — a
client can request any subset (J3-J6, drag, SRP, 3rd-body) via the API and
the integrator wires them in without internal branching.

Conventions
-----------
- t : TDB seconds since J2000 (matches SPICE ET).
- r, v : Cartesian position (m), velocity (m/s) in the body-centred inertial
  frame whose Z axis is the body's rotation pole (Earth GCRS / J2000 for
  Earth; ICRF for other bodies once we wire them up).
- All vectors are NumPy 1-D float arrays of length 3.

References
----------
- Vallado 4th ed. §8 for zonal harmonics.
- Montenbruck & Gill, "Satellite Orbits" §3 for third-body, drag, SRP.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from oamp.bodies import (
    ASTRONOMICAL_UNIT_M,
    EARTH,
    SOLAR_RADIATION_PRESSURE_AU,
    Body,
)


# Vehicle parameters needed by drag and SRP. Mass is required, the rest have
# sensible CubeSat-sized defaults so callers can opt in incrementally.
@dataclass(frozen=True, slots=True)
class Vehicle:
    mass_kg: float
    drag_area_m2: float = 1.0
    drag_cd: float = 2.2  # typical for a tumbling spacecraft
    srp_area_m2: float = 1.0
    srp_cr: float = 1.5  # 1.0 = fully absorbing, 2.0 = perfect reflector


class Perturbation(Protocol):
    """A perturbation contributes an acceleration vector each RHS evaluation."""

    def __call__(self, t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
#  Zonal harmonics  J_n  (n = 2..6)
# --------------------------------------------------------------------------- #
#
# Geopotential convention (Vallado §8.6):
#
#     U(r) = -(μ/r) [1 - Σ_n J_n (R/r)^n P_n(sin φ)]
#          = -μ/r + μ J_n R^n / r^(n+1) · P_n(u)
#
# where u = sin φ = z/r. The disturbing-potential term V_n = +(μ J_n R^n /
# r^(n+1)) P_n(u) gives the acceleration a = −∇V_n, whose Cartesian components
# are computed below from a single closed-form expression in terms of P_n and
# P_n':
#
#     a_x = +(μ J_n R^n / r^(n+3)) · x · [(n+1) P_n(u) + u P_n'(u)]
#     a_y = +(μ J_n R^n / r^(n+3)) · y · [(n+1) P_n(u) + u P_n'(u)]
#     a_z = +(μ J_n R^n / r^(n+3)) · [(n+1) z P_n(u) − r (1−u²) P_n'(u)]
#
# This avoids the per-degree closed-form Cartesian expansions where it is easy
# to lose a sign. The formula is validated against the pure J2 Cartesian form
# in the unit tests.


# Legendre polynomials P_n(u) and their derivatives P_n'(u) for n = 2..6.
def _legendre_p_and_dp(n: int, u: float) -> tuple[float, float]:
    if n == 2:
        return 0.5 * (3 * u * u - 1), 3 * u
    if n == 3:
        return 0.5 * u * (5 * u * u - 3), 0.5 * (15 * u * u - 3)
    if n == 4:
        u2 = u * u
        return (35 * u2 * u2 - 30 * u2 + 3) / 8, (140 * u2 * u - 60 * u) / 8
    if n == 5:
        u2 = u * u
        u3 = u2 * u
        u5 = u2 * u3
        return (63 * u5 - 70 * u3 + 15 * u) / 8, (315 * u2 * u2 - 210 * u2 + 15) / 8
    if n == 6:
        u2 = u * u
        u4 = u2 * u2
        u6 = u2 * u4
        return (
            (231 * u6 - 315 * u4 + 105 * u2 - 5) / 16,
            (1386 * u4 * u - 1260 * u2 * u + 210 * u) / 16,
        )
    raise ValueError(f"Legendre P_{n} not implemented")


def _accel_jn(n: int, r: np.ndarray, mu: float, R: float, Jn: float) -> np.ndarray:
    """Acceleration from a single zonal harmonic J_n."""
    x, y, z = r
    rn = float(np.linalg.norm(r))
    u = z / rn
    Pn, dPn = _legendre_p_and_dp(n, u)
    common_xy = (n + 1) * Pn + u * dPn
    coeff = mu * Jn * R**n / rn ** (n + 3)
    a_xy = coeff * common_xy
    a_z_factor = (n + 1) * z * Pn - rn * (1 - u * u) * dPn
    return np.array([a_xy * x, a_xy * y, coeff * a_z_factor])


def zonal_harmonics(body: Body, n_max: int) -> Perturbation:
    """Zonal harmonics J2..J{n_max}. n_max=2 for J2 only, n_max=6 for the full set."""
    if n_max < 2:
        raise ValueError("n_max must be >= 2 (J2 minimum)")
    if n_max > 6:
        raise ValueError(f"zonal harmonics above J6 not implemented (got {n_max})")
    coeffs = body.jn[: n_max - 1]  # jn[0]=J2, jn[1]=J3, ...

    def _accel(_t: float, r: np.ndarray, _v: np.ndarray) -> np.ndarray:
        total = np.zeros(3)
        for i, Jn in enumerate(coeffs):
            if Jn == 0.0:
                continue
            total = total + _accel_jn(i + 2, r, body.mu, body.radius, Jn)
        return total

    return _accel


# --------------------------------------------------------------------------- #
#  Third-body gravity  (Sun, Moon, planets)
# --------------------------------------------------------------------------- #


def third_body(
    target_name: str,
    target_mu: float,
    observer: str = "EARTH",
    frame: str = "J2000",
) -> Perturbation:
    """Indirect + direct point-mass perturbation from a third body.

    Uses Battin's regularised expression to avoid catastrophic cancellation
    between the direct and indirect terms when |r_sat| ≪ |r_body|:

        a = μ_k  ·  [ -(r + s) f(q)  −  r ] / |s|^3

    with q = (r·(r + 2 s)) / |s|², s = r_body − r_observer, and
    f(q) = q(3 + 3q + q²) / (1 + (1+q)^{3/2}).

    The SPICE call is the only step that touches I/O — for performance it
    could be cached, but at 1 ephemeris call per RHS evaluation it's cheap.
    """

    def _accel(t: float, r: np.ndarray, _v: np.ndarray) -> np.ndarray:
        from oamp import spice  # local import: SPICE is optional

        s, _ = spice.body_state(target_name, t, observer=observer, frame=frame)
        s = np.asarray(s, dtype=float)
        d = r - s  # vector from third body to spacecraft
        d_norm = float(np.linalg.norm(d))
        s_norm = float(np.linalg.norm(s))
        if d_norm == 0.0 or s_norm == 0.0:
            return np.zeros(3)
        # Direct + indirect, regularised (Battin §8.6.4 / Montenbruck eq. 3.36).
        q = float(np.dot(r, r - 2 * s) / np.dot(s, s))
        fq = q * (3 + 3 * q + q * q) / (1 + (1 + q) ** 1.5)
        return -target_mu * (r + fq * s) / s_norm**3

    return _accel


# --------------------------------------------------------------------------- #
#  Solar radiation pressure  (cannonball, with cylindrical Earth shadow)
# --------------------------------------------------------------------------- #


def solar_radiation_pressure(
    vehicle: Vehicle,
    observer: str = "EARTH",
    frame: str = "J2000",
    body_radius_m: float = EARTH.radius,
) -> Perturbation:
    """Cannonball SRP with cylindrical Earth-shadow eclipse.

    a_SRP = ν · (P0 · (AU/r_sun)²) · C_R · A/m · r̂_sun→sat

    where ν ∈ [0, 1] is the shadow function. The cylindrical model is
    coarse (no penumbra) but adequate for LEO/MEO trade studies.
    """

    def _accel(t: float, r: np.ndarray, _v: np.ndarray) -> np.ndarray:
        from oamp import spice

        r_sun, _ = spice.body_state("SUN", t, observer=observer, frame=frame)
        r_sun = np.asarray(r_sun, dtype=float)

        # Vector from satellite to Sun.
        sat_to_sun = r_sun - r
        d_au = float(np.linalg.norm(sat_to_sun)) / ASTRONOMICAL_UNIT_M
        if d_au == 0.0:
            return np.zeros(3)
        sun_hat = sat_to_sun / np.linalg.norm(sat_to_sun)

        # Cylindrical-shadow occultation: spacecraft is in eclipse when
        # (r · -sun_hat) > 0  AND  perp distance < body_radius.
        anti_sun = -sun_hat
        along = float(np.dot(r, anti_sun))
        if along > 0.0:
            perp = r - along * anti_sun
            if float(np.linalg.norm(perp)) < body_radius_m:
                return np.zeros(3)  # in shadow

        P = SOLAR_RADIATION_PRESSURE_AU / (d_au * d_au)
        coeff = -P * vehicle.srp_cr * vehicle.srp_area_m2 / vehicle.mass_kg
        # Force is along the photon direction (sun → sat = -sun_hat).
        return coeff * sun_hat

    return _accel


# --------------------------------------------------------------------------- #
#  Atmospheric drag  (exponential atmosphere; NRLMSISE optional)
# --------------------------------------------------------------------------- #


# Crude US Standard Atmosphere fit — good to ~factor of 2 from 0 to 1000 km.
# When the user installs `nrlmsise00` we'll swap this for a date-aware model;
# the exponential fit suffices for early integration tests and CI.
_RHO_0 = 1.225  # kg/m³ sea-level
_H_BASE = 8500.0  # scale height, m


def _exponential_density(altitude_m: float) -> float:
    if altitude_m < 0:
        return _RHO_0
    if altitude_m > 1.5e6:
        return 0.0
    # Two-segment fit: dense low-atmosphere + thin upper-atmosphere.
    if altitude_m < 100_000.0:
        return _RHO_0 * math.exp(-altitude_m / _H_BASE)
    # Above 100 km, scale height grows from ~6 km to ~60 km. Use a piecewise
    # linear interpolation of log(rho) anchored at MSIS-derived points.
    table = [
        (100_000.0, 5.604e-7),
        (150_000.0, 2.076e-9),
        (200_000.0, 2.541e-10),
        (300_000.0, 1.916e-11),
        (400_000.0, 2.803e-12),
        (500_000.0, 5.215e-13),
        (700_000.0, 3.560e-14),
        (1_000_000.0, 3.561e-15),
    ]
    for (h0, rho0), (h1, rho1) in zip(table, table[1:], strict=False):
        if h0 <= altitude_m <= h1:
            f = (altitude_m - h0) / (h1 - h0)
            return math.exp(math.log(rho0) + f * (math.log(rho1) - math.log(rho0)))
    return 0.0


def atmospheric_drag(
    vehicle: Vehicle,
    body: Body = EARTH,
    density_fn: Callable[[float], float] | None = None,
    density_fn_full: Callable[[float, np.ndarray], float] | None = None,
) -> Perturbation:
    """Drag against an atmosphere that co-rotates with the central body.

    a_drag = −½ ρ Cd A/m |v_rel| v_rel,   v_rel = v − ω × r

    Two density-function flavours are supported:

    * ``density_fn(altitude_m) -> rho`` for altitude-only models (the default
      exponential fit).
    * ``density_fn_full(t_tdb, r_inertial) -> rho`` for models that depend on
      epoch and geographic position — e.g. NRLMSISE / MSIS via
      :func:`msis_density_fn`.

    Only one of the two may be supplied.
    """
    if density_fn is not None and density_fn_full is not None:
        raise ValueError("supply at most one of density_fn / density_fn_full")
    rho_simple = density_fn or _exponential_density
    rho_full = density_fn_full

    def _accel(t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        altitude = float(np.linalg.norm(r)) - body.radius
        d = rho_full(t, r) if rho_full is not None else rho_simple(altitude)
        if d == 0.0:
            return np.zeros(3)
        # Inertial → relative velocity via co-rotation: ω × r.
        omega = np.array([0.0, 0.0, body.omega])
        v_rel = v - np.cross(omega, r)
        speed = float(np.linalg.norm(v_rel))
        if speed == 0.0:
            return np.zeros(3)
        coeff = -0.5 * d * vehicle.drag_cd * vehicle.drag_area_m2 / vehicle.mass_kg
        return coeff * speed * v_rel

    return _accel


# --------------------------------------------------------------------------- #
#  Composite RHS builder
# --------------------------------------------------------------------------- #


def compose(*perturbations: Perturbation) -> Perturbation:
    """Sum any number of perturbation accelerations into one."""

    def _sum(t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        total = np.zeros(3)
        for p in perturbations:
            total = total + p(t, r, v)
        return total

    return _sum
