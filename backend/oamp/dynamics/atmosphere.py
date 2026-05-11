"""Atmospheric density models that need more than just altitude.

The exponential / piecewise fit in :mod:`oamp.dynamics.perturbations` is good
enough for trade-study CI. When the user opts into MSIS (NRLMSISE-00 / MSIS-2)
via the ``pymsis`` package, this module provides a density function that
captures the diurnal bulge, latitudinal variation, and solar-flux dependence
that drive the bulk of LEO orbit-decay uncertainty.

The dependence on ``pymsis`` is lazy: callers import the factory but don't pay
the import cost until they actually exercise the model. If ``pymsis`` is not
installed, the factory raises ``ImportError`` with a clear remediation.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from oamp.bodies import EARTH

# Earth's angular speed * t maps inertial frame X-axis to ECEF X-axis (very
# coarse — ignores precession, nutation, polar motion). For drag-model
# fidelity that's well below MSIS's own ~30% accuracy, this is fine.
_OMEGA_EARTH = EARTH.omega


def _epoch_to_datetime64(t_tdb: float) -> np.datetime64:
    """Convert TDB seconds since J2000 to numpy datetime64 (UTC, ignoring leap
    seconds — MSIS only cares about UT1-ish accuracy)."""
    # J2000.0 epoch: 2000-01-01T11:58:55.816 UTC. We approximate as 12:00 UTC.
    j2000 = np.datetime64("2000-01-01T12:00:00")
    return j2000 + np.timedelta64(int(t_tdb * 1e6), "us")


def _eci_to_lat_lon(r: np.ndarray, t_tdb: float) -> tuple[float, float, float]:
    """ECI → (lat_deg, lon_deg, alt_km). Lon is wrapped to ±180°."""
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    rn = math.sqrt(x * x + y * y + z * z)
    lat = math.degrees(math.asin(z / rn))
    # Rough rotation: GMST ≈ ω_Earth · t since J2000 (ignoring epoch offset).
    gmst = _OMEGA_EARTH * t_tdb
    lon_inertial = math.atan2(y, x)
    lon = math.degrees((lon_inertial - gmst + math.pi) % (2 * math.pi) - math.pi)
    alt_km = (rn - EARTH.radius) / 1000.0
    return lat, lon, alt_km


def msis_density_fn(
    *,
    f107: float | None = None,
    f107a: float | None = None,
    ap: float | None = None,
) -> Callable[[float, np.ndarray], float]:
    """Build a (t, r) → ρ callable backed by pymsis (MSIS-2 by default).

    Solar/geomagnetic inputs default to climatology if not supplied. The
    returned callable is safe to invoke from inside an integrator's RHS — each
    call dispatches one pymsis run for the given (time, lat, lon, altitude).

    Raises ``ImportError`` if pymsis is not installed.
    """
    try:
        from pymsis import msis
    except ImportError as e:
        raise ImportError(
            "MSIS atmospheric model requires the 'pymsis' package. "
            "Install via `pixi add --pypi pymsis`."
        ) from e

    def _rho(t: float, r: np.ndarray) -> float:
        lat, lon, alt_km = _eci_to_lat_lon(r, t)
        if alt_km < 0 or alt_km > 1500.0:
            return 0.0
        date = _epoch_to_datetime64(t)
        kwargs: dict = {}
        if f107 is not None:
            kwargs["f107"] = f107
        if f107a is not None:
            kwargs["f107a"] = f107a
        if ap is not None:
            kwargs["ap"] = ap
        out = msis.run(date, lon, lat, alt_km, **kwargs)
        # Output is shape (1, 11): index 0 is total mass density kg/m³.
        rho = float(np.asarray(out)[0, 0])
        if not np.isfinite(rho) or rho < 0:
            return 0.0
        return rho

    return _rho
