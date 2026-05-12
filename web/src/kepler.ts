// Two-body Keplerian element ↔ Cartesian conversion for the editor's
// "Elements" mode. Single-frame, no perturbations — purely a UX helper so the
// user can edit a, e, i, Ω, ω, ν instead of raw r/v. Propagation still happens
// on the backend with the resulting Cartesian state.
//
// Conventions: classical elements (a in metres, e dimensionless, i/Ω/ω/ν in
// radians internally; UI uses km and degrees). All angles are equatorial /
// J2000-aligned with the central body. For e = 1 (parabolic) the conversion is
// undefined; callers should clamp e away from 1.

export type OrbitalElements = {
  a:    number;  // semi-major axis (m); negative for hyperbolae
  e:    number;  // eccentricity (≥ 0)
  i:    number;  // inclination (rad)
  raan: number;  // right-ascension of ascending node Ω (rad)
  argp: number;  // argument of periapsis ω (rad)
  nu:   number;  // true anomaly ν (rad)
};

type Vec3 = [number, number, number];

const TWO_PI = 2 * Math.PI;

function norm3(v: Vec3): number { return Math.hypot(v[0], v[1], v[2]); }
function dot3(a: Vec3, b: Vec3): number { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function cross3(a: Vec3, b: Vec3): Vec3 {
  return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]];
}
function wrap2pi(x: number): number {
  let y = x % TWO_PI;
  if (y < 0) y += TWO_PI;
  return y;
}

/** Convert Cartesian state (r, v in m, m/s) to classical orbital elements. */
export function stateToElements(r: Vec3, v: Vec3, mu: number): OrbitalElements {
  const rmag = norm3(r);
  const vmag = norm3(v);
  if (rmag === 0) throw new Error("zero radius");

  // Specific angular momentum h = r × v.
  const h = cross3(r, v);
  const hmag = norm3(h);

  // Node vector n = ẑ × h (points to ascending node).
  const n: Vec3 = [-h[1], h[0], 0];
  const nmag = Math.hypot(n[0], n[1]);

  // Eccentricity vector e = (v × h)/μ − r̂.
  const vxh = cross3(v, h);
  const eVec: Vec3 = [
    vxh[0]/mu - r[0]/rmag,
    vxh[1]/mu - r[1]/rmag,
    vxh[2]/mu - r[2]/rmag,
  ];
  const e = norm3(eVec);

  // Specific orbital energy ε = v²/2 − μ/r → a = −μ/(2ε).
  const eps = (vmag * vmag) / 2 - mu / rmag;
  const a = Math.abs(eps) < 1e-12 ? Infinity : -mu / (2 * eps);

  // Inclination from h_z / |h|.
  const i = Math.acos(Math.max(-1, Math.min(1, h[2] / hmag)));

  // RAAN: angle in ECI x–y plane to the node line.
  let raan = 0;
  if (nmag > 1e-12) {
    raan = Math.acos(Math.max(-1, Math.min(1, n[0] / nmag)));
    if (n[1] < 0) raan = TWO_PI - raan;
  }

  // Argument of periapsis ω: angle from n̂ to ê in the orbital plane.
  let argp = 0;
  if (nmag > 1e-12 && e > 1e-12) {
    argp = Math.acos(Math.max(-1, Math.min(1, dot3(n, eVec) / (nmag * e))));
    if (eVec[2] < 0) argp = TWO_PI - argp;
  } else if (e > 1e-12) {
    // Equatorial, elliptic — argp degenerates with raan; pick the longitude of
    // periapsis (ω̃) and stash it in argp with raan = 0.
    argp = Math.atan2(eVec[1], eVec[0]);
    if (h[2] < 0) argp = TWO_PI - argp;
  }

  // True anomaly ν: angle from ê to r̂.
  let nu = 0;
  if (e > 1e-12) {
    nu = Math.acos(Math.max(-1, Math.min(1, dot3(eVec, r) / (e * rmag))));
    if (dot3(r, v) < 0) nu = TWO_PI - nu;
  } else {
    // Circular: use argument of latitude (or true longitude for equatorial).
    if (nmag > 1e-12) {
      nu = Math.acos(Math.max(-1, Math.min(1, dot3(n, r) / (nmag * rmag))));
      if (r[2] < 0) nu = TWO_PI - nu;
    } else {
      nu = Math.atan2(r[1], r[0]);
      if (h[2] < 0) nu = TWO_PI - nu;
    }
  }

  return {
    a,
    e,
    i,
    raan: wrap2pi(raan),
    argp: wrap2pi(argp),
    nu:   wrap2pi(nu),
  };
}

/** Convert classical elements to Cartesian state (m, m/s). */
export function elementsToState(el: OrbitalElements, mu: number): { r: Vec3; v: Vec3 } {
  const { a, e, i, raan, argp, nu } = el;
  if (Math.abs(1 - e) < 1e-12) throw new Error("parabolic orbit (e ≈ 1) not supported");

  // Semi-latus rectum p = a(1 − e²); valid for hyperbolae with a < 0 too.
  const p = a * (1 - e * e);
  const cosNu = Math.cos(nu);
  const sinNu = Math.sin(nu);
  const r_pf = p / (1 + e * cosNu);  // radius in perifocal frame

  // Perifocal: x along periapsis, y along velocity at periapsis (in-plane), z out.
  const r_perif: Vec3 = [r_pf * cosNu, r_pf * sinNu, 0];
  const k = Math.sqrt(mu / p);
  const v_perif: Vec3 = [-k * sinNu, k * (e + cosNu), 0];

  // Rotation perifocal → inertial: R3(−Ω) R1(−i) R3(−ω).
  const cR = Math.cos(raan), sR = Math.sin(raan);
  const cI = Math.cos(i),    sI = Math.sin(i);
  const cP = Math.cos(argp), sP = Math.sin(argp);

  // Combined rotation matrix R (3×3).
  const R: number[][] = [
    [ cR*cP - sR*sP*cI, -cR*sP - sR*cP*cI,  sR*sI ],
    [ sR*cP + cR*sP*cI, -sR*sP + cR*cP*cI, -cR*sI ],
    [ sP*sI,             cP*sI,              cI    ],
  ];
  const apply = (m: number[][], v: Vec3): Vec3 => [
    m[0]![0]!*v[0] + m[0]![1]!*v[1] + m[0]![2]!*v[2],
    m[1]![0]!*v[0] + m[1]![1]!*v[1] + m[1]![2]!*v[2],
    m[2]![0]!*v[0] + m[2]![1]!*v[1] + m[2]![2]!*v[2],
  ];
  return { r: apply(R, r_perif), v: apply(R, v_perif) };
}

export const RAD = Math.PI / 180;
export const DEG = 180 / Math.PI;
