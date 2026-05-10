"""Time-scale handling for OAMP.

Convention (locked in for the project):
- **Internal**: ephemeris time (ET) == TDB seconds since J2000 TDB epoch.
  This is what SPICE uses and what every dynamics function consumes.
- **Wire**: UTC ISO 8601 strings (`"2026-05-10T12:00:00Z"`) for all REST/WS
  payloads. Conversion happens at the API edge, never inside dynamics code.

This module provides the conversions. When SPICE is available with an LSK
kernel loaded it delegates to SpiceyPy (full-fidelity, leap-second correct).
Otherwise it falls back to a fixed leap-second table that's good through
2026-12-31 — a `LeapSecondError` is raised if a date outside that range is
queried so failures are loud, not silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

# TT - TAI is fixed by definition.
TT_MINUS_TAI: Final[float] = 32.184

# TAI - UTC at the J2000 epoch (2000-01-01).
TAI_MINUS_UTC_AT_J2000: Final[int] = 32

# J2000.0 = 2000-01-01T12:00:00 TT  (== 2000-01-01T11:58:55.816 UTC).
J2000_UTC: Final[datetime] = datetime(2000, 1, 1, 11, 58, 55, 816_000, tzinfo=UTC)

# TAI-UTC (leap seconds). Each entry is (UTC datetime when this offset began,
# offset in seconds). Source: IERS Bulletin C; valid through end of 2026.
_LEAP_TABLE: Final[tuple[tuple[datetime, int], ...]] = (
    (datetime(1972, 1, 1, tzinfo=UTC), 10),
    (datetime(1972, 7, 1, tzinfo=UTC), 11),
    (datetime(1973, 1, 1, tzinfo=UTC), 12),
    (datetime(1974, 1, 1, tzinfo=UTC), 13),
    (datetime(1975, 1, 1, tzinfo=UTC), 14),
    (datetime(1976, 1, 1, tzinfo=UTC), 15),
    (datetime(1977, 1, 1, tzinfo=UTC), 16),
    (datetime(1978, 1, 1, tzinfo=UTC), 17),
    (datetime(1979, 1, 1, tzinfo=UTC), 18),
    (datetime(1980, 1, 1, tzinfo=UTC), 19),
    (datetime(1981, 7, 1, tzinfo=UTC), 20),
    (datetime(1982, 7, 1, tzinfo=UTC), 21),
    (datetime(1983, 7, 1, tzinfo=UTC), 22),
    (datetime(1985, 7, 1, tzinfo=UTC), 23),
    (datetime(1988, 1, 1, tzinfo=UTC), 24),
    (datetime(1990, 1, 1, tzinfo=UTC), 25),
    (datetime(1991, 1, 1, tzinfo=UTC), 26),
    (datetime(1992, 7, 1, tzinfo=UTC), 27),
    (datetime(1993, 7, 1, tzinfo=UTC), 28),
    (datetime(1994, 7, 1, tzinfo=UTC), 29),
    (datetime(1996, 1, 1, tzinfo=UTC), 30),
    (datetime(1997, 7, 1, tzinfo=UTC), 31),
    (datetime(1999, 1, 1, tzinfo=UTC), 32),
    (datetime(2006, 1, 1, tzinfo=UTC), 33),
    (datetime(2009, 1, 1, tzinfo=UTC), 34),
    (datetime(2012, 7, 1, tzinfo=UTC), 35),
    (datetime(2015, 7, 1, tzinfo=UTC), 36),
    (datetime(2017, 1, 1, tzinfo=UTC), 37),
)
_LEAP_VALID_UNTIL: Final[datetime] = datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC)


class LeapSecondError(ValueError):
    """Raised when a UTC instant falls outside the embedded leap-second table.

    Once raised, either update `_LEAP_TABLE` or load a SPICE LSK kernel.
    """


def _tai_minus_utc_seconds(utc: datetime) -> int:
    if utc < _LEAP_TABLE[0][0]:
        raise LeapSecondError(f"UTC {utc.isoformat()} is before TAI-UTC was defined (1972)")
    if utc > _LEAP_VALID_UNTIL:
        raise LeapSecondError(
            f"UTC {utc.isoformat()} is past the embedded leap-second table "
            f"(valid through {_LEAP_VALID_UNTIL.date()}). "
            "Load a SPICE LSK kernel or update _LEAP_TABLE."
        )
    offset = _LEAP_TABLE[0][1]
    for boundary, value in _LEAP_TABLE:
        if utc >= boundary:
            offset = value
        else:
            break
    return offset


@dataclass(frozen=True, slots=True)
class Instant:
    """An instant in time, carrying its scale explicitly.

    Stored as TDB seconds since J2000 (== SPICE ET). UTC representation is
    derived on demand.
    """

    et: float  # TDB seconds since J2000

    @classmethod
    def from_utc_iso(cls, iso: str) -> Instant:
        return cls(et=utc_iso_to_et(iso))

    def to_utc_iso(self) -> str:
        return et_to_utc_iso(self.et)


def _try_spice_et(iso: str) -> float | None:
    """Use SPICE if an LSK kernel is loaded. Returns None on any failure."""
    try:
        import spiceypy as sp
    except ImportError:
        return None
    try:
        return float(sp.str2et(iso))
    except Exception:
        return None


def utc_iso_to_et(iso: str) -> float:
    """Parse a UTC ISO 8601 string and return ET (TDB seconds since J2000).

    Uses SPICE if an LSK is loaded; otherwise falls back to the embedded leap
    table. The fallback ignores TDB-TT periodic terms (≤ 1.6 ms), which is
    fine for everything except sub-millisecond mission timing.
    """
    et = _try_spice_et(iso)
    if et is not None:
        return et

    # Normalize "Z" suffix and missing timezone into a tz-aware UTC datetime.
    s = iso.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt_utc = dt.astimezone(UTC)

    # ET ≈ TT seconds since J2000
    #     = (UTC delta from J2000_UTC) + [(TAI-UTC) at t] - [(TAI-UTC) at J2000]
    # The TT-TAI = 32.184 s offset cancels (it's the same at both endpoints),
    # leaving only the leap-second drift since J2000. TDB-TT periodic terms
    # (≤ 1.6 ms) are ignored in the fallback path.
    delta_utc = (dt_utc - J2000_UTC).total_seconds()
    tai_minus_utc = _tai_minus_utc_seconds(dt_utc)
    return delta_utc + (tai_minus_utc - TAI_MINUS_UTC_AT_J2000)


def et_to_utc_iso(et: float) -> str:
    """Inverse of `utc_iso_to_et`. Microsecond precision."""
    try:
        import spiceypy as sp

        # "ISOC" = ISO Calendar (UTC). 6 decimals == microseconds.
        result = sp.timout(et, "YYYY-MM-DDTHR:MN:SC.######", 64)
        return result.strip() + "Z"
    except Exception:
        # Inverse of the fallback in utc_iso_to_et:
        #     delta_utc = et - (TAI-UTC(t) - TAI-UTC(J2000))
        # TAI-UTC depends on the UTC instant we're solving for, but it only
        # changes at integer-second boundaries — two fixed-point iterations
        # are more than enough.
        approx = J2000_UTC.timestamp() + et + TAI_MINUS_UTC_AT_J2000
        for _ in range(2):
            dt_guess = datetime.fromtimestamp(approx, tz=UTC)
            tai_minus_utc = _tai_minus_utc_seconds(dt_guess)
            approx = J2000_UTC.timestamp() + et - (tai_minus_utc - TAI_MINUS_UTC_AT_J2000)
        dt = datetime.fromtimestamp(approx, tz=UTC)
        return dt.isoformat().replace("+00:00", "Z")
