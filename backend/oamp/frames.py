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
