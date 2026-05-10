"""Central-body constants for OAMP.

All values in SI (m, s, kg). Sources:
- IAU 2015 / NIST CODATA 2018 for fundamental constants.
- DE440 README for GM (mu) values.
- WGS-84 for Earth's equatorial radius.
- EGM2008 for Earth's J2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Body:
    name: str
    mu: float       # gravitational parameter, m^3 / s^2
    radius: float   # equatorial radius, m
    j2: float       # second zonal harmonic (dimensionless)


EARTH: Final[Body] = Body(
    name="Earth",
    mu=3.986004418e14,
    radius=6_378_137.0,
    j2=1.082626173852223e-3,
)

SUN: Final[Body] = Body(
    name="Sun",
    mu=1.32712440041939e20,
    radius=6.957e8,
    j2=2.198e-7,
)

MOON: Final[Body] = Body(
    name="Moon",
    mu=4.9048695e12,
    radius=1_737_400.0,
    j2=2.0321e-4,
)
