"""Central-body constants for OAMP.

All values in SI (m, s, kg). Sources:
- IAU 2015 / NIST CODATA 2018 for fundamental constants.
- DE440 README for GM (mu) values.
- WGS-84 for Earth's equatorial radius and reciprocal flattening.
- EGM2008 (unnormalised) for Earth's zonal harmonics J2–J6.
- IAU Working Group 2009 (Archinal et al.) for body pole orientation in J2000.

Frame conventions
-----------------
The default propagation frame is J2000 / EME2000 — equatorial mean of
J2000.0 epoch.  By construction Earth's mean rotation pole *is* +Z in this
frame, so Earth has ``pole_ra_j2000 = 0``, ``pole_dec_j2000 = π/2``.  Other
bodies (Moon, Sun, …) have poles offset from +Z; for them the (RA, Dec)
values below define the pole direction in J2000 inertial coordinates, and
``prime_meridian_w0_j2000`` is the rotation angle of the body's prime
meridian about its own pole at the J2000 epoch.  These are needed for
zonal-harmonic perturbations on non-Earth central bodies: J_n is symmetric
about the body's spin axis, *not* about J2000 +Z.

Shape: ``radius`` is the equatorial radius; ``polar_radius`` is along the
spin axis.  ``inv_flattening`` = R_eq / (R_eq − R_pol) is provided as a
convenience and matches the standard cartographic value (WGS-84 = 298.257).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final


@dataclass(frozen=True, slots=True)
class Body:
    name: str
    mu: float  # gravitational parameter, m^3 / s^2
    radius: float  # equatorial radius, m
    j2: float  # second zonal harmonic (dimensionless)
    # Higher-order zonal harmonics (J3..J6 are the practical limit before
    # tesseral terms dominate for LEO precision orbits).
    jn: tuple[float, float, float, float, float, float] = field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    # Sidereal rotation rate, rad/s. Used by the drag model to derive the
    # co-rotating wind (v_rel = v_inertial − ω × r) and by the IAU prime-
    # meridian model: W(t) = W₀ + ω · (t − t₀_J2000).
    omega: float = 0.0
    # ---- Shape ---------------------------------------------------------- #
    # Polar radius along the spin axis. Defaults to `radius` (sphere) when
    # not specified.
    polar_radius: float = 0.0
    # Reciprocal flattening 1/f = R_eq / (R_eq − R_pol). 0 means "sphere".
    inv_flattening: float = 0.0
    # ---- Pole orientation in J2000 ------------------------------------- #
    # IAU 2009 right ascension of the spin pole (rad). Earth's pole is +Z
    # in J2000 *by definition* of the frame, so RA = 0, Dec = π/2.
    pole_ra_j2000: float = 0.0
    pole_dec_j2000: float = math.pi / 2
    # Prime-meridian angle W at the J2000 epoch (rad). For Earth this is
    # GMST at J2000 = ~4.8950 rad; for the Moon it's ~38.32° = 0.6687 rad
    # (Cassini state). For the Sun: ~84.176° = 1.4693 rad.
    prime_meridian_w0_j2000: float = 0.0


# EGM2008 unnormalised zonal coefficients for Earth (J2..J7).
# Reference: Pavlis et al. 2012, "The development and evaluation of EGM2008".
_EARTH_JN: Final[tuple[float, float, float, float, float, float]] = (
    1.082626173852223e-3,  # J2  (also exposed as Body.j2 for back-compat)
    -2.532613e-6,  # J3
    -1.619898e-6,  # J4
    -2.277345e-7,  # J5
    5.406653e-7,  # J6
    -3.523609e-7,  # J7
)

# WGS-84: R_eq = 6 378 137.0 m, 1/f = 298.257_223_563 → R_pol ≈ 6 356 752.314 m.
_EARTH_R_EQ: Final[float] = 6_378_137.0
_EARTH_INV_F: Final[float] = 298.257_223_563
_EARTH_R_POL: Final[float] = _EARTH_R_EQ * (1.0 - 1.0 / _EARTH_INV_F)

# GMST at J2000 epoch — mirrors `oamp.frames.GMST_AT_J2000_RAD`. Duplicated
# here as a Body constant so `Body.prime_meridian_w0_j2000` is meaningful
# without importing `frames`.
_GMST_AT_J2000_RAD: Final[float] = 4.894961212823058

EARTH: Final[Body] = Body(
    name="Earth",
    mu=3.986004418e14,
    radius=_EARTH_R_EQ,
    j2=_EARTH_JN[0],
    jn=_EARTH_JN,
    omega=7.2921150e-5,
    polar_radius=_EARTH_R_POL,
    inv_flattening=_EARTH_INV_F,
    pole_ra_j2000=0.0,
    pole_dec_j2000=math.pi / 2,
    prime_meridian_w0_j2000=_GMST_AT_J2000_RAD,
)

# Sun: nearly spherical (1/f ≈ 1/9e-6 ≈ 1e5 — treat as sphere).
# Pole: IAU α₀ = 286.13°, δ₀ = 63.87°; W₀ = 84.176°.
SUN: Final[Body] = Body(
    name="Sun",
    mu=1.32712440041939e20,
    radius=6.957e8,
    j2=2.198e-7,
    omega=2.865329e-6,  # 25.38-day sidereal at the equator
    polar_radius=6.957e8,
    inv_flattening=0.0,
    pole_ra_j2000=math.radians(286.13),
    pole_dec_j2000=math.radians(63.87),
    prime_meridian_w0_j2000=math.radians(84.176),
)

# Lunar unnormalised zonal harmonics (GRGM1200A low-order truncation).
# J2 is the dominant term; J3..J6 are an order of magnitude or two smaller
# than Earth's but well-determined.  We include them so jn_max≥2 actually
# yields a contribution when MOON is the central body.
_MOON_JN: Final[tuple[float, float, float, float, float, float]] = (
    2.032e-4,    # J2
    8.475e-6,    # J3
    -9.592e-6,   # J4
    7.158e-7,    # J5
    -1.357e-5,   # J6
    0.0,         # J7 (not in low-order LP solutions)
)

# Moon (IAU 2009 mean values):
#   α₀ = 269.9949°, δ₀ = 66.5392°, W₀ = 38.3213°
#   R = 1 737.4 km (mean), polar 1 736.0 km → 1/f ≈ 1241
MOON: Final[Body] = Body(
    name="Moon",
    mu=4.9048695e12,
    radius=1_737_400.0,
    j2=_MOON_JN[0],
    jn=_MOON_JN,
    omega=2.6617e-6,  # synchronous rotation with orbital motion
    polar_radius=1_736_000.0,
    inv_flattening=1240.6,
    pole_ra_j2000=math.radians(269.9949),
    pole_dec_j2000=math.radians(66.5392),
    prime_meridian_w0_j2000=math.radians(38.3213),
)


# Solar constants used by the SRP model. Source: NASA Solar System Fact Sheet.
SOLAR_RADIATION_PRESSURE_AU: Final[float] = 4.56e-6  # N / m^2 at 1 AU
ASTRONOMICAL_UNIT_M: Final[float] = 1.495_978_707e11  # m

# Speed of light, used by SRP and (later) PN/GR corrections.
SPEED_OF_LIGHT: Final[float] = 299_792_458.0  # m/s, exact (SI definition)

# Mean obliquity of the ecliptic at J2000.0 — angle between Earth's equator
# (XY plane of J2000) and the ecliptic plane (Earth's mean orbital plane).
# IAU 2006 value. Constant on the timescale of any single mission; precesses
# at ~46.8 arcsec/century from external lunisolar torques.
OBLIQUITY_J2000_RAD: Final[float] = math.radians(23.439_291_111)
