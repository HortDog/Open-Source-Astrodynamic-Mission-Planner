"""Two-Line-Element (TLE) ingest.

Parses the standard NORAD TLE format, runs SGP4 to propagate to a requested
UTC instant, and returns an inertial state suitable for the rest of OAMP.

SGP4 produces position/velocity in the True Equator Mean Equinox (TEME) frame
of date. For Phase 2's visualization-fidelity needs (≲ km accuracy at LEO)
TEME is treated as equivalent to J2000 / ECI — the rotation between them is
small (< 1 km/yr) and the bigger error in TLE ephemerides is the SGP4 model
itself, not the frame.

A higher-fidelity TEME → ICRF transformation (precession/nutation via IAU 2006
or SPICE FK kernel) is a Phase 3 concern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import httpx
from sgp4.api import Satrec, jday

from oamp.bodies import EARTH

KM_TO_M = 1000.0
KMPS_TO_MPS = 1000.0

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"


@dataclass(frozen=True, slots=True)
class TLEState:
    name: str
    norad_id: int
    epoch_jd: float  # Julian day of TLE epoch (UTC)
    r_m: tuple[float, float, float]  # ECI position (m, TEME ≈ J2000)
    v_m_s: tuple[float, float, float]  # ECI velocity (m/s)
    altitude_km: float
    period_minutes: float


def parse_tle(line1: str, line2: str, name: str = "") -> Satrec:
    """Parse a two-line element set into an SGP4 propagator object."""
    line1, line2 = line1.strip(), line2.strip()
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError(
            f"TLE format error: line1 must start with '1 ', line2 with '2 '"
            f" (got {line1[:2]!r}, {line2[:2]!r})"
        )
    if len(line1) < 69 or len(line2) < 69:
        raise ValueError(f"TLE lines too short ({len(line1)}, {len(line2)}); expected 69 chars")
    return Satrec.twoline2rv(line1, line2)


def propagate_tle(
    satrec: Satrec,
    jd: float | None = None,
    fr: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Run SGP4 at the given Julian day. Defaults to the TLE epoch.

    Returns (r_m, v_m_s, jd_used). Raises if SGP4 reports an error code.
    """
    if jd is None:
        jd, fr = satrec.jdsatepoch, satrec.jdsatepochF
    err, r_km, v_km_s = satrec.sgp4(jd, fr)
    if err != 0:
        # SGP4 error codes documented in Vallado/Hoots 2006.
        raise RuntimeError(f"SGP4 propagation failed: error code {err}")
    r = (r_km[0] * KM_TO_M, r_km[1] * KM_TO_M, r_km[2] * KM_TO_M)
    v = (v_km_s[0] * KMPS_TO_MPS, v_km_s[1] * KMPS_TO_MPS, v_km_s[2] * KMPS_TO_MPS)
    return r, v, jd + fr


def tle_state(
    line1: str,
    line2: str,
    name: str = "",
    norad_id: int | None = None,
) -> TLEState:
    """Build a TLEState at the element-set epoch."""
    sat = parse_tle(line1, line2, name)
    r, v, jd = propagate_tle(sat)
    altitude_km = (math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2) - EARTH.radius) / 1000.0
    # Period from mean motion (rev/day → seconds).
    period_min = (2 * math.pi / sat.no_kozai) if sat.no_kozai > 0 else math.inf
    return TLEState(
        name=name.strip(),
        norad_id=int(norad_id if norad_id is not None else sat.satnum),
        epoch_jd=jd,
        r_m=r,
        v_m_s=v,
        altitude_km=altitude_km,
        period_minutes=period_min,
    )


def fetch_celestrak(norad_id: int, timeout_s: float = 10.0) -> tuple[str, str, str]:
    """Fetch a TLE from Celestrak by NORAD catalogue number.

    Returns (name, line1, line2). Raises on HTTP error or malformed response.
    """
    params = {"CATNR": str(int(norad_id)), "FORMAT": "TLE"}
    resp = httpx.get(CELESTRAK_URL, params=params, timeout=timeout_s)
    resp.raise_for_status()
    lines = [ln for ln in resp.text.splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(
            f"Celestrak returned {len(lines)} non-empty lines for NORAD {norad_id} "
            f"(expected 3): {resp.text!r}"
        )
    return lines[0].strip(), lines[1], lines[2]


def jd_from_iso_utc(utc: str) -> tuple[float, float]:
    """Convert an ISO-8601 UTC string to (jd, fr) for the SGP4 API.

    Accepts ``YYYY-MM-DDTHH:MM:SS`` with optional fractional seconds. We don't
    deal with leap seconds — SGP4 only needs UT1-ish accuracy.
    """
    from datetime import datetime

    s = utc.rstrip("Z")
    dt = datetime.fromisoformat(s)
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)
    return jd, fr
