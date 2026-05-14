"""Reference-frame transforms (Phase 4).

Phase 4 introduces the Earth–Moon synodic frame, which rotates with the
Earth–Moon line so the two primaries appear stationary.  The instantaneous
rotation matrix and angular velocity are derived from SPICE-supplied Moon
state vectors, then applied to a spacecraft state.

The default convention places the origin at the **Earth–Moon barycenter**
(matching CR3BP usage in `oamp.dynamics.cr3bp`).  Spacecraft state vectors
passed in are assumed to be Earth-centric inertial J2000 (BCI_EARTH); the
inverse transform restores that convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from oamp.dynamics.cr3bp import EM_MU


class Frame(StrEnum):
    """Reference frames recognised by the propagator and renderer."""

    J2000_EARTH = "J2000_EARTH"   # inertial, Earth-centric (default)
    EM_SYNODIC = "EM_SYNODIC"     # Earth–Moon barycentric rotating (SI units)
    ECEF = "ECEF"                 # Earth-centred Earth-fixed (rotates at ω⊕)


# --------------------------------------------------------------------------- #
#  Greenwich Mean Sidereal Time + ECEF ↔ J2000.
#
#  The constants and convention mirror the front-end `web/src/render/gmst.ts`
#  so both sides compute the same θ_G for a given TDB epoch — important for
#  round-tripping ECEF initial conditions submitted by the editor.
# --------------------------------------------------------------------------- #

EARTH_OMEGA_RAD_S: float = 7.2921158553e-5
GMST_AT_J2000_RAD: float = 4.894961212823058

EARTH_OMEGA_VEC: np.ndarray = np.array([0.0, 0.0, EARTH_OMEGA_RAD_S], dtype=float)


def gmst_from_tdb(t_tdb_s: float) -> float:
    """Greenwich Mean Sidereal Time in radians given TDB seconds since J2000.

    Constant-rate model. Accuracy at the editor's zoom band is well sub-degree;
    for sub-arcsecond geodesy use the IAU-2006/2000 series instead.
    """
    if not np.isfinite(t_tdb_s):
        return 0.0
    theta = GMST_AT_J2000_RAD + EARTH_OMEGA_RAD_S * float(t_tdb_s)
    # Wrap into [0, 2π) to keep downstream math well-conditioned for long arcs.
    return float(theta % (2.0 * np.pi))


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def j2000_to_ecef(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    t_tdb: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate an inertial Earth-centric state into the Earth-fixed frame.

        r_ECEF = R_z(−θ_G) · r_ECI
        v_ECEF = R_z(−θ_G) · v_ECI − ω⊕ × r_ECEF
    """
    theta = gmst_from_tdb(t_tdb)
    R = _rot_z(-theta)
    r_ecef = R @ np.asarray(r_eci, dtype=float)
    v_rot = R @ np.asarray(v_eci, dtype=float)
    v_ecef = v_rot - np.cross(EARTH_OMEGA_VEC, r_ecef)
    return r_ecef, v_ecef


def ecef_to_j2000(
    r_ecef: np.ndarray,
    v_ecef: np.ndarray,
    t_tdb: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `j2000_to_ecef` — Earth-fixed state → inertial Earth-centric.

        r_ECI = R_z(θ_G) · r_ECEF
        v_ECI = R_z(θ_G) · (v_ECEF + ω⊕ × r_ECEF)
    """
    theta = gmst_from_tdb(t_tdb)
    R = _rot_z(theta)
    r = np.asarray(r_ecef, dtype=float)
    v = np.asarray(v_ecef, dtype=float) + np.cross(EARTH_OMEGA_VEC, r)
    return R @ r, R @ v


# --------------------------------------------------------------------------- #
#  Body-pole orientation in J2000.
#
#  Each non-Earth body (Moon, Sun, …) has a spin pole offset from J2000 +Z.
#  Zonal harmonics J_n are symmetric about the *body's* spin axis, so a
#  spacecraft propagated around the Moon needs r rotated into a pole-aligned
#  frame before the J_n acceleration is evaluated. The helpers below build
#  that rotation from `Body.pole_ra_j2000` / `Body.pole_dec_j2000`.
#
#  Earth's pole IS J2000 +Z by definition, so the rotation is the identity
#  and the existing two-line `_accel_jn` produces the same result as before.
# --------------------------------------------------------------------------- #


def body_pole_in_j2000(body) -> np.ndarray:
    """Unit vector along the body's spin axis in J2000 coordinates."""
    a = float(body.pole_ra_j2000)
    d = float(body.pole_dec_j2000)
    return np.array(
        [np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)],
        dtype=float,
    )


def pole_alignment_rotation(pole_j2000: np.ndarray) -> np.ndarray:
    """3×3 rotation that takes a J2000 vector into a frame where the body's
    pole is along +Z. The inverse (transpose) maps back to J2000.

    Rodrigues' formula. Identity when `pole_j2000` is already +Z.
    """
    p = np.asarray(pole_j2000, dtype=float)
    p = p / np.linalg.norm(p)
    z = np.array([0.0, 0.0, 1.0])
    cos_a = float(np.dot(p, z))
    if cos_a > 1.0 - 1e-15:
        return np.eye(3)
    if cos_a < -1.0 + 1e-15:
        # Antiparallel: 180° flip about any axis perpendicular to z.
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(p, z)
    axis = axis / np.linalg.norm(axis)
    sin_a = float(np.sqrt(max(0.0, 1.0 - cos_a * cos_a)))
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)


def body_pole_alignment_rotation(body) -> np.ndarray:
    """Rotation that takes a J2000 vector into the body's pole-aligned frame.

    Cached pattern: equivalent to ``pole_alignment_rotation(body_pole_in_j2000(body))``
    but exposed as a single call for callers that don't need the pole vector.
    """
    return pole_alignment_rotation(body_pole_in_j2000(body))


@dataclass(frozen=True, slots=True)
class EmSynodicBasis:
    """Instantaneous Earth–Moon synodic basis at a TDB epoch."""

    R_eci_to_syn: np.ndarray   # 3×3 rotation: ECI → synodic
    omega_eci: np.ndarray      # synodic-frame angular velocity in ECI (rad/s)
    d_em_m: float              # instantaneous Earth–Moon distance (m)


def em_synodic_basis(t_tdb: float) -> EmSynodicBasis:
    """Build the Earth-centric Earth–Moon synodic basis at TDB epoch (s).

    Requires SPICE kernels.  The basis is defined by the *instantaneous* Moon
    state — so the frame is not strictly uniform-rotation (the Moon's true
    orbit has e≈0.055), but is the standard practical convention.
    """
    from oamp import spice  # local import — SPICE is optional

    r_moon, v_moon = spice.body_state("MOON", t_tdb, "EARTH", "J2000")
    r_moon = np.asarray(r_moon, dtype=float)
    v_moon = np.asarray(v_moon, dtype=float)

    d_em = float(np.linalg.norm(r_moon))
    x_hat = r_moon / d_em
    h = np.cross(r_moon, v_moon)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)
    # Rotation matrix: rows are the synodic basis expressed in ECI.
    R = np.vstack((x_hat, y_hat, z_hat))
    # Angular velocity vector (in ECI): ω = h / r² along z_hat.
    omega = h / (d_em * d_em)
    return EmSynodicBasis(R_eci_to_syn=R, omega_eci=omega, d_em_m=d_em)


def inertial_to_em_synodic(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    t_tdb: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform an Earth-centric J2000 state to barycentric Earth–Moon synodic.

    Returns ``(r_syn, v_syn)`` in SI units (m, m/s).  The synodic frame origin
    is the Earth–Moon barycenter — i.e. Earth lies at ``(−μ d_em, 0, 0)`` and
    the Moon at ``((1−μ) d_em, 0, 0)``.
    """
    b = em_synodic_basis(t_tdb)
    # Rotate to synodic axes (still Earth-centric).
    r_rot = b.R_eci_to_syn @ np.asarray(r_eci, dtype=float)
    v_rot = b.R_eci_to_syn @ np.asarray(v_eci, dtype=float)
    # Subtract the frame's rotational velocity: v_rot_frame = v_inertial − ω × r
    omega_syn = b.R_eci_to_syn @ b.omega_eci
    v_rot = v_rot - np.cross(omega_syn, r_rot)
    # Translate origin Earth → barycenter (barycenter sits at (μ d_em, 0, 0)
    # along +x from Earth).
    r_bary = r_rot - np.array([EM_MU * b.d_em_m, 0.0, 0.0])
    return r_bary, v_rot


def em_synodic_to_inertial(
    r_syn: np.ndarray,
    v_syn: np.ndarray,
    t_tdb: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `inertial_to_em_synodic` — synodic SI → Earth-centric ECI."""
    b = em_synodic_basis(t_tdb)
    # Translate origin barycenter → Earth (shift along +x by μ d_em).
    r_rot = np.asarray(r_syn, dtype=float) + np.array([EM_MU * b.d_em_m, 0.0, 0.0])
    omega_syn = b.R_eci_to_syn @ b.omega_eci
    v_rot_inertial = np.asarray(v_syn, dtype=float) + np.cross(omega_syn, r_rot)
    R_inv = b.R_eci_to_syn.T  # rotation is orthogonal
    return R_inv @ r_rot, R_inv @ v_rot_inertial
