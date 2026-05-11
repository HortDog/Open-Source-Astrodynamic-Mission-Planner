"""Central-body constants for OAMP.

All values in SI (m, s, kg). Sources:
- IAU 2015 / NIST CODATA 2018 for fundamental constants.
- DE440 README for GM (mu) values.
- WGS-84 for Earth's equatorial radius.
- EGM2008 (unnormalised) for Earth's zonal harmonics J2–J6.
"""

from __future__ import annotations

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
    # co-rotating wind (v_rel = v_inertial − ω × r).
    omega: float = 0.0


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

EARTH: Final[Body] = Body(
    name="Earth",
    mu=3.986004418e14,
    radius=6_378_137.0,
    j2=_EARTH_JN[0],
    jn=_EARTH_JN,
    omega=7.2921150e-5,
)

SUN: Final[Body] = Body(
    name="Sun",
    mu=1.32712440041939e20,
    radius=6.957e8,
    j2=2.198e-7,
    omega=2.865329e-6,  # 25.38-day sidereal at the equator
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

MOON: Final[Body] = Body(
    name="Moon",
    mu=4.9048695e12,
    radius=1_737_400.0,
    j2=_MOON_JN[0],
    jn=_MOON_JN,
    omega=2.6617e-6,  # synchronous rotation with orbital motion
)


# Solar constants used by the SRP model. Source: NASA Solar System Fact Sheet.
SOLAR_RADIATION_PRESSURE_AU: Final[float] = 4.56e-6  # N / m^2 at 1 AU
ASTRONOMICAL_UNIT_M: Final[float] = 1.495_978_707e11  # m

# Speed of light, used by SRP and (later) PN/GR corrections.
SPEED_OF_LIGHT: Final[float] = 299_792_458.0  # m/s, exact (SI definition)
