// Greenwich Mean Sidereal Time — the rotation angle of Earth's prime
// meridian relative to the vernal equinox direction (≈ inertial +X in J2000).
//
// For mission-planner visualisation we only need accuracy to about a degree;
// the constant-rate approximation below is good to << 1° per century, which
// is well below the wireframe-line thickness on screen.
//
// Reference values (IAU 1982 / Vallado §3.5):
//   ω⊕      = 7.2921158553 × 10⁻⁵ rad/s  (sidereal rotation rate)
//   GMST(J2000) = 18h 41m 50.5s = 280.4606°
//                = 4.894961212823058 rad
//
// We treat TDB ≈ UTC for the angle — the actual TDB−UTC offset (≤ 70 s)
// corresponds to ≤ 0.3° of rotation, invisible at the editor's zoom band.

export const EARTH_OMEGA_RAD_S = 7.2921158553e-5;
export const GMST_AT_J2000_RAD = 4.894961212823058;

const TWO_PI = 2 * Math.PI;

/** Greenwich angle θ_G in radians (mod 2π) given TDB seconds since J2000. */
export function gmstFromTdb(t_tdb_s: number): number {
  if (!Number.isFinite(t_tdb_s)) return 0;
  const theta = GMST_AT_J2000_RAD + EARTH_OMEGA_RAD_S * t_tdb_s;
  // Wrap into [0, 2π) so rendering math stays well-conditioned at long arcs.
  let w = theta % TWO_PI;
  if (w < 0) w += TWO_PI;
  return w;
}
