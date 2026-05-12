import {
  cr3bpLagrange,
  cr3bpManifold,
  cr3bpPeriodicOrbit,
  cr3bpPropagate,
  cr3bpWsb,
  fetchHealth,
  optimizeHohmann,
  optimizeLambert,
  optimizeMultiBurn,
  propagate,
  runLaunch,
  SimSocket,
  spiceEphemeris,
  spiceState,
  tleByNorad,
  tleParse,
  transformStates,
  type FiniteBurnSpec,
  type Vec3,
} from "./api";
import { initRenderer, type Renderer } from "./render/scene";

const status = document.getElementById("status")!;
const version = document.getElementById("version")!;
const banner = document.getElementById("banner")!;
const canvas = document.getElementById("scene") as HTMLCanvasElement;
const btnLeo = document.getElementById("btn-leo") as HTMLButtonElement;
const btnLaunch = document.getElementById("btn-launch") as HTMLButtonElement;
const btnLeoJ2 = document.getElementById("btn-leo-j2") as HTMLButtonElement;
const btnHohmann = document.getElementById("btn-hohmann") as HTMLButtonElement;
const btnLambert = document.getElementById("btn-lambert") as HTMLButtonElement;
const btnCislunar = document.getElementById("btn-cislunar") as HTMLButtonElement;
const btnCr3bp = document.getElementById("btn-cr3bp") as HTMLButtonElement;
const btnManifold = document.getElementById("btn-manifold") as HTMLButtonElement;
const btnWsb = document.getElementById("btn-wsb") as HTMLButtonElement;
const btnEditor = document.getElementById("btn-editor") as HTMLButtonElement;
const editor = document.getElementById("editor") as HTMLElement;
const scrub = document.getElementById("scrub") as HTMLInputElement;
const scrubInfo = document.getElementById("scrub-info") as HTMLSpanElement;
const camTargetSel = document.getElementById("cam-target") as HTMLSelectElement;
const allButtons = [
  btnLeo, btnLaunch, btnLeoJ2, btnHohmann, btnLambert,
  btnCislunar, btnCr3bp, btnManifold, btnWsb,
];

const setStatus = (s: string) => { status.textContent = s; };
const showBanner = (msg: string) => { banner.textContent = msg; banner.classList.add("show"); };

const MU_EARTH = 3.986004418e14;
const R_EARTH = 6_378_137;
const R0 = 7_000_000;
const INCLINATION_DEG = 51.6;

// Body constants table for the central-body selector (frontend mirror of the
// canonical values in oamp.bodies — used only to seed the editor IC defaults).
type BodyName = "EARTH" | "MOON";
const BODY: Record<BodyName, { mu: number; radius: number; default_alt_m: number; label: string }> = {
  EARTH: { mu: 3.986004418e14, radius: 6_378_137,  default_alt_m: 622_000,  label: "Earth" },
  MOON:  { mu: 4.9048695e12,   radius: 1_737_400,  default_alt_m: 100_000,  label: "Moon"  },
};

// Earth--Moon system constants used by the EM_SYNODIC view-frame transform.
// Mirror of backend `oamp.dynamics.cr3bp.EM_MU` / `EM_LENGTH_M`.
const EM_MU_FRONTEND = BODY.MOON.mu / (BODY.EARTH.mu + BODY.MOON.mu);
const EM_LENGTH_M = 384_400_000;

// Phase colours (phosphor palette + accents for distinct transfer phases).
const C_DEPART:  [number, number, number] = [0.20, 1.00, 0.30]; // bright phosphor — initial orbit
const C_TRANSFER:[number, number, number] = [1.00, 0.70, 0.10]; // amber — transfer arc
const C_ARRIVE:  [number, number, number] = [0.30, 0.70, 1.00]; // cyan — destination orbit
const C_BURN:    [number, number, number] = [1.00, 0.18, 0.08]; // phosphor red — finite burn

function selectButton(active: HTMLButtonElement): void {
  for (const b of allButtons) {
    b.setAttribute("aria-pressed", b === active ? "true" : "false");
  }
}

// --------------------------------------------------------------------------- //
//  Scrubber: track the active trajectory so the slider can position the marker.
// --------------------------------------------------------------------------- //

type ActiveTrajectory = { t: number[]; states: number[][] };
let activeTraj: ActiveTrajectory | null = null;
let activeRenderer: Renderer | null = null;
let latestMoonR: Vec3 | null = null;
let latestCraftR: Vec3 = [0, 0, 0];
// Decimated Moon track over the current trajectory.  Sampled at `moonTrackT`
// (seconds from t0); positions in inertial m relative to Earth.
let moonTrackT: number[] | null = null;
let moonTrackR: Vec3[] | null = null;

function refreshCameraTarget(): void {
  if (!activeRenderer) return;
  const choice = camTargetSel.value;
  if (choice === "craft") {
    activeRenderer.setCameraTarget(latestCraftR);
  } else if (choice === "moon" && latestMoonR) {
    activeRenderer.setCameraTarget(latestMoonR);
  } else {
    activeRenderer.setCameraTarget(null);
  }
}

camTargetSel.addEventListener("change", refreshCameraTarget);

/** Find apsides: zero-crossings of the radial velocity ṙ = (r·v)/|r|.
 *  More robust than triplet-extremum detection — won't pepper a near-circular
 *  orbit with markers from floating-point wobble.  Returns nothing for orbits
 *  whose radial span is below ~0.1% of the mean (effectively circular). */
function findApses(
  states: number[][],
): Array<{ position: Vec3; isPeriapsis: boolean }> {
  const out: Array<{ position: Vec3; isPeriapsis: boolean }> = [];
  if (states.length < 3) return out;
  // Skip markers when the orbit is too circular to have a meaningful apse.
  let rMin = Infinity, rMax = 0;
  for (const s of states) {
    const r = Math.hypot(s[0]!, s[1]!, s[2]!);
    if (r < rMin) rMin = r;
    if (r > rMax) rMax = r;
  }
  if (rMax - rMin < 1e-3 * (rMax + rMin) * 0.5) return out;

  const rdot = (s: number[]) => {
    const r = Math.hypot(s[0]!, s[1]!, s[2]!);
    return (s[0]! * s[3]! + s[1]! * s[4]! + s[2]! * s[5]!) / r;
  };
  let prev = rdot(states[0]!);
  for (let i = 1; i < states.length; i++) {
    const cur = rdot(states[i]!);
    const s = states[i]!;
    // ṙ goes negative → positive : periapsis just crossed.
    // ṙ goes positive → negative : apoapsis just crossed.
    if (prev < 0 && cur >= 0) {
      out.push({ position: [s[0]!, s[1]!, s[2]!], isPeriapsis: true });
    } else if (prev > 0 && cur <= 0) {
      out.push({ position: [s[0]!, s[1]!, s[2]!], isPeriapsis: false });
    }
    prev = cur;
  }
  return out;
}

const ALT_W = 200, ALT_H = 30;
let altMinR = 0, altMaxR = 1;

/** Plot |r|(t) into the footer SVG.  X spans the full trajectory; Y spans
 *  [r_min, r_max] padded by 5% so the apsides aren't flush against the edges. */
function drawAltitudeChart(times: number[], states: number[][]): void {
  const line = document.getElementById("alt-line") as unknown as SVGPolylineElement;
  if (!line || states.length < 2) {
    if (line) line.setAttribute("points", "");
    return;
  }
  const rs = states.map((s) => Math.hypot(s[0]!, s[1]!, s[2]!));
  altMinR = Math.min(...rs);
  altMaxR = Math.max(...rs);
  const pad = Math.max((altMaxR - altMinR) * 0.05, 1);
  const yLo = altMinR - pad, yHi = altMaxR + pad;
  const t0 = times[0]!, tN = times[times.length - 1]!;
  const tSpan = Math.max(tN - t0, 1);
  const pts: string[] = new Array(rs.length);
  for (let i = 0; i < rs.length; i++) {
    const x = ((times[i]! - t0) / tSpan) * ALT_W;
    const y = ALT_H - ((rs[i]! - yLo) / (yHi - yLo)) * ALT_H;
    pts[i] = `${x.toFixed(1)},${y.toFixed(1)}`;
  }
  line.setAttribute("points", pts.join(" "));
}

function updateAltitudeCursor(idx: number): void {
  if (!activeTraj) return;
  const cursor = document.getElementById("alt-cursor") as unknown as SVGLineElement;
  const label = document.getElementById("alt-label");
  if (!cursor) return;
  const N = activeTraj.t.length;
  if (N < 2) return;
  const t0 = activeTraj.t[0]!, tN = activeTraj.t[N - 1]!;
  const tSpan = Math.max(tN - t0, 1);
  const x = ((activeTraj.t[idx]! - t0) / tSpan) * ALT_W;
  cursor.setAttribute("x1", x.toFixed(1));
  cursor.setAttribute("x2", x.toFixed(1));
  if (label) {
    label.textContent =
      `|r|: ${(altMinR / 1000).toFixed(0)}–${(altMaxR / 1000).toFixed(0)} km`;
  }
}

function setActiveTrajectory(t: number[], states: number[][]): void {
  activeTraj = { t, states };
  scrub.min = "0";
  scrub.max = String(Math.max(0, states.length - 1));
  scrub.value = "0";
  if (activeRenderer) {
    activeRenderer.setApses(findApses(states));
    // Lagrange-point markers and manifold tubes are scenario-specific; clear
    // on every trajectory load.  Demos re-set them right after this call.
    activeRenderer.setLagrangePoints([]);
    activeRenderer.setManifoldTubes([]);
  }
  drawAltitudeChart(t, states);
  updateScrubMarker(0);
}

function updateScrubMarker(idx: number): void {
  if (!activeTraj || !activeRenderer) return;
  const s = activeTraj.states[idx];
  const t = activeTraj.t[idx];
  if (!s || t === undefined) return;
  latestCraftR = [s[0]!, s[1]!, s[2]!];
  activeRenderer.setMarker(
    latestCraftR,
    s.length >= 6 ? [s[3]!, s[4]!, s[5]!] : null,
  );
  if (camTargetSel.value === "craft") activeRenderer.setCameraTarget(latestCraftR);
  // Animated Moon: pick the precomputed sample nearest to the current time.
  if (moonTrackT && moonTrackR && moonTrackR.length > 0) {
    let lo = 0, hi = moonTrackT.length - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (moonTrackT[mid]! <= t) lo = mid; else hi = mid;
    }
    const pick = (t - moonTrackT[lo]!) < (moonTrackT[hi]! - t) ? lo : hi;
    latestMoonR = moonTrackR[pick]!;
    activeRenderer.setSecondaryBody(latestMoonR, BODY.MOON.radius);
    if (camTargetSel.value === "moon") activeRenderer.setCameraTarget(latestMoonR);
  }
  const r = Math.hypot(s[0]!, s[1]!, s[2]!);
  const v = Math.hypot(s[3]!, s[4]!, s[5]!);
  scrubInfo.textContent =
    `t=${t.toFixed(0)} s · |r|=${(r / 1000).toFixed(0)} km · |v|=${(v / 1000).toFixed(2)} km/s`;
  updateAltitudeCursor(idx);
}

scrub.addEventListener("input", () => updateScrubMarker(parseInt(scrub.value, 10)));

function statesToPositions(states: number[][]): Float32Array<ArrayBuffer> {
  const out = new Float32Array(states.length * 3);
  for (let i = 0; i < states.length; i++) {
    const s = states[i]!;
    out[i * 3 + 0] = s[0]!;
    out[i * 3 + 1] = s[1]!;
    out[i * 3 + 2] = s[2]!;
  }
  return out;
}

/** Override colours for samples that fall inside a finite-burn window.
 *  Mutates `buf` in-place and returns it. */
function applyBurnWindows(
  buf: Float32Array<ArrayBuffer>,
  times: number[],
  burns: Array<[number, number]>,   // [t_start, t_end] per burn
): Float32Array<ArrayBuffer> {
  for (let i = 0; i < times.length; i++) {
    const t = times[i]!;
    for (const [t0, t1] of burns) {
      if (t >= t0 && t <= t1) {
        buf[i * 3 + 0] = C_BURN[0];
        buf[i * 3 + 1] = C_BURN[1];
        buf[i * 3 + 2] = C_BURN[2];
        break;
      }
    }
  }
  return buf;
}

/** Build a per-vertex colour buffer that paints each segment of the trajectory
 *  in one of N phase colours. `phaseStarts` is the time threshold at which to
 *  switch to the next colour; `times` is the per-sample timestamp array. */
function colorByPhases(
  times: number[],
  phaseStarts: number[],
  colors: ReadonlyArray<[number, number, number]>,
): Float32Array<ArrayBuffer> {
  const N = times.length;
  const out = new Float32Array(N * 3);
  let phase = 0;
  for (let i = 0; i < N; i++) {
    while (phase + 1 < phaseStarts.length && times[i]! >= phaseStarts[phase + 1]!) {
      phase++;
    }
    const c = colors[Math.min(phase, colors.length - 1)]!;
    out[i * 3 + 0] = c[0];
    out[i * 3 + 1] = c[1];
    out[i * 3 + 2] = c[2];
  }
  return out;
}

async function showLeo(renderer: Renderer, j2 = false): Promise<void> {
  if (renderer.kind !== "webgpu") return;
  renderer.setSceneScale(R0);
  renderer.setCentralBody(R_EARTH);
  const v0 = Math.sqrt(MU_EARTH / R0);
  const i = (INCLINATION_DEG * Math.PI) / 180;
  const period = 2 * Math.PI * Math.sqrt((R0 * R0 * R0) / MU_EARTH);
  const duration = j2 ? 8 * period : period;
  const steps = j2 ? 1600 : 400;

  setStatus(`propagating LEO${j2 ? " + J2 (8 orbits)" : ""}…`);
  const data = await propagate({
    state: { r: [R0, 0, 0], v: [0, v0 * Math.cos(i), v0 * Math.sin(i)] },
    duration_s: duration,
    steps,
    mu: MU_EARTH,
    j2_enabled: j2,
    body_radius: R_EARTH,
  });
  renderer.drawTrajectory(statesToPositions(data.states));
  setActiveTrajectory(data.t, data.states);
  setStatus(j2
    ? `LEO + J2: ${data.states.length} samples over ${(duration / 60).toFixed(0)} min — note nodal regression`
    : `LEO: ${data.states.length} samples over one period (${period.toFixed(0)} s)`);
}

async function showLaunch(renderer: Renderer): Promise<void> {
  if (renderer.kind !== "webgpu") return;
  renderer.setSceneScale(R_EARTH);
  renderer.setCentralBody(R_EARTH);

  setStatus("running launch sim…");
  const data = await runLaunch();
  const positions = statesToPositions(data.states);

  const N = data.states.length;
  const colors = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    let r = 1.0, g = 0.55, b = 0.18;
    if (i > data.burnout_index) { r = 1.0; g = 0.92; b = 0.45; }
    if (i > data.circularization_index) { r = 0.55; g = 0.85; b = 1.0; }
    colors[i * 3 + 0] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  renderer.drawTrajectory(positions, colors);
  setActiveTrajectory(data.t, data.states);
  setStatus(
    `Launch demo: burnout T+${data.burnout_time_s.toFixed(0)}s, ` +
    `Δv_circ=${data.circularization_dv_m_s.toFixed(0)} m/s, ` +
    `final orbit ${data.final_periapsis_km.toFixed(0)} × ${data.final_apoapsis_km.toFixed(0)} km`,
  );
}

/** Hohmann LEO→GEO. Three phases displayed:
 *  green (1 LEO orbit) → amber (half transfer ellipse) → cyan (1 GEO orbit). */
async function showHohmann(renderer: Renderer): Promise<void> {
  if (renderer.kind !== "webgpu") return;

  const r_leo = R_EARTH + 400e3;       // 400 km LEO
  const r_geo = 42_164e3;              // GEO

  renderer.setSceneScale(r_geo);
  renderer.setCentralBody(R_EARTH);

  setStatus("solving Hohmann transfer…");
  const ho = await optimizeHohmann({ r1_m: r_leo, r2_m: r_geo, mu: MU_EARTH });

  // 1 LEO period for context + half-transfer + 1 GEO period.
  const T_leo  = 2 * Math.PI * Math.sqrt(r_leo ** 3 / MU_EARTH);
  const T_geo  = 2 * Math.PI * Math.sqrt(r_geo ** 3 / MU_EARTH);
  const t_dv1  = T_leo;
  const t_dv2  = T_leo + ho.transfer_time_s;
  const t_end  = t_dv2 + T_geo;
  const v_leo  = Math.sqrt(MU_EARTH / r_leo);

  // Start at (r_leo, 0, 0) moving in +Y. Prograde RIC = +in-track.
  const data = await propagate({
    state: { r: [r_leo, 0, 0], v: [0, v_leo, 0] },
    duration_s: t_end,
    steps: 1200,
    mu: MU_EARTH,
    body_radius: R_EARTH,
    maneuvers: [
      { t_offset_s: t_dv1, dv_ric: [0, ho.dv1_m_s, 0] },
      { t_offset_s: t_dv2, dv_ric: [0, ho.dv2_m_s, 0] },
    ],
  });

  const positions = statesToPositions(data.states);
  const colors = colorByPhases(
    data.t,
    [0, t_dv1, t_dv2],
    [C_DEPART, C_TRANSFER, C_ARRIVE],
  );
  renderer.drawTrajectory(positions, colors);
  setActiveTrajectory(data.t, data.states);

  setStatus(
    `Hohmann LEO(400 km)→GEO  ` +
    `Δv₁=${(ho.dv1_m_s / 1000).toFixed(2)} km/s · ` +
    `Δv₂=${(ho.dv2_m_s / 1000).toFixed(2)} km/s · ` +
    `Δv_total=${(ho.dv_total_m_s / 1000).toFixed(2)} km/s · ` +
    `TOF=${(ho.transfer_time_s / 3600).toFixed(2)} h`,
  );
}

/** Lambert intercept: depart from a 500 km circular orbit, arrive at a target
 *  point on a 4 000 km circular orbit 120° downrange after a chosen TOF. */
async function showLambert(renderer: Renderer): Promise<void> {
  if (renderer.kind !== "webgpu") return;

  const r_dep = R_EARTH + 500e3;
  const r_arr = R_EARTH + 4_000e3;
  const transfer_angle_deg = 120;
  const tof_s = 4 * 60 * 60;          // 4 h

  renderer.setSceneScale(r_arr * 1.2);
  renderer.setCentralBody(R_EARTH);

  setStatus("solving Lambert problem…");

  const cos_a = Math.cos((transfer_angle_deg * Math.PI) / 180);
  const sin_a = Math.sin((transfer_angle_deg * Math.PI) / 180);
  const r1: Vec3 = [r_dep, 0, 0];
  const r2: Vec3 = [r_arr * cos_a, r_arr * sin_a, 0];

  const lb = await optimizeLambert({
    r1_m: r1,
    r2_m: r2,
    tof_s,
    mu: MU_EARTH,
  });
  if (!lb.converged) { setStatus("Lambert did not converge"); return; }

  // Velocities on each circular orbit (departure + arrival) — needed to size
  // the Δv arrows and to extend the visible orbit for context.
  const v_dep_circ = Math.sqrt(MU_EARTH / r_dep);
  const v_arr_circ = Math.sqrt(MU_EARTH / r_arr);
  // Circular velocity at the arrival point (perpendicular to radial in +z plane).
  const v_arr_dir: Vec3 = [-Math.sin((transfer_angle_deg * Math.PI) / 180) * v_arr_circ,
                            Math.cos((transfer_angle_deg * Math.PI) / 180) * v_arr_circ,
                            0];

  // Δv at departure: lambert.v1 − circular velocity at r1 = (0, v_dep_circ, 0).
  const dv1: Vec3 = [lb.v1_m_s[0], lb.v1_m_s[1] - v_dep_circ, lb.v1_m_s[2]];
  // Δv at arrival: circular velocity at r2 − lambert.v2.
  const dv2: Vec3 = [v_arr_dir[0] - lb.v2_m_s[0],
                     v_arr_dir[1] - lb.v2_m_s[1],
                     v_arr_dir[2] - lb.v2_m_s[2]];

  // Phase 1: half of the departure circle for context.
  const T_dep = 2 * Math.PI * Math.sqrt(r_dep ** 3 / MU_EARTH);
  const T_arr = 2 * Math.PI * Math.sqrt(r_arr ** 3 / MU_EARTH);
  // Total span: 1/2 departure orbit + transfer + 1/2 arrival orbit.
  const t_dv1 = T_dep * 0.5;
  const t_dv2 = t_dv1 + tof_s;
  const t_end = t_dv2 + T_arr * 0.5;

  // Convert inertial Δv → RIC at the manoeuvre instant.
  // At t_dv1 the spacecraft is on a circular orbit through r1 with v = (0, v_dep_circ, 0)
  // returning to (r_dep, 0, 0). At that point R̂=+x, Î=+y, Ĉ=+z.
  const dv1_ric: Vec3 = [dv1[0], dv1[1], dv1[2]];
  // At t_dv2 the spacecraft is at r2 with Lambert velocity v2. R̂=r2/|r2|, Î=Ĉ×R̂.
  const r2n = Math.hypot(r2[0], r2[1], r2[2]);
  const r_hat: Vec3 = [r2[0] / r2n, r2[1] / r2n, r2[2] / r2n];
  // Both r2 and the velocity lie in the xy plane → Ĉ = +Z. Î = Ĉ × R̂ is the
  // in-plane perpendicular rotated 90° CCW from R̂.
  const c_hat: Vec3 = [0, 0, 1];
  const i_hat: Vec3 = [
    c_hat[1] * r_hat[2] - c_hat[2] * r_hat[1],
    c_hat[2] * r_hat[0] - c_hat[0] * r_hat[2],
    c_hat[0] * r_hat[1] - c_hat[1] * r_hat[0],
  ];
  const dv2_ric: Vec3 = [
    dv2[0] * r_hat[0] + dv2[1] * r_hat[1] + dv2[2] * r_hat[2],
    dv2[0] * i_hat[0] + dv2[1] * i_hat[1] + dv2[2] * i_hat[2],
    dv2[0] * c_hat[0] + dv2[1] * c_hat[1] + dv2[2] * c_hat[2],
  ];

  const data = await propagate({
    state: { r: [r_dep, 0, 0], v: [0, v_dep_circ, 0] },
    duration_s: t_end,
    steps: 1200,
    mu: MU_EARTH,
    body_radius: R_EARTH,
    maneuvers: [
      { t_offset_s: t_dv1, dv_ric: dv1_ric },
      { t_offset_s: t_dv2, dv_ric: dv2_ric },
    ],
  });

  const positions = statesToPositions(data.states);
  const colors = colorByPhases(
    data.t,
    [0, t_dv1, t_dv2],
    [C_DEPART, C_TRANSFER, C_ARRIVE],
  );
  renderer.drawTrajectory(positions, colors);
  setActiveTrajectory(data.t, data.states);

  const dv1_mag = Math.hypot(dv1[0], dv1[1], dv1[2]);
  const dv2_mag = Math.hypot(dv2[0], dv2[1], dv2[2]);
  setStatus(
    `Lambert 500→4000 km, Δν=${transfer_angle_deg}°, TOF=${(tof_s / 3600).toFixed(1)}h · ` +
    `Δv₁=${(dv1_mag / 1000).toFixed(2)} km/s · ` +
    `Δv₂=${(dv2_mag / 1000).toFixed(2)} km/s · ` +
    `Δv_total=${((dv1_mag + dv2_mag) / 1000).toFixed(2)} km/s · ` +
    `solver: ${lb.converged ? "converged" : "diverged"}`,
  );
}

/** Cislunar demo: Earth-Moon Lambert transfer.
 *
 *  Loads a 200-km LEO initial state, queries the Moon's ephemeris at a fixed
 *  demo epoch via SPICE, solves Lambert for a ~3.5-day transfer to that
 *  position, applies the TLI Δv as an impulsive RIC maneuver, propagates with
 *  J2 + third-body Moon active, and renders the Moon at its inertial position.
 *  Requires SPICE kernels — bail with a hint if `/spice/state` is unavailable. */
async function showCislunar(renderer: Renderer): Promise<void> {
  if (renderer.kind !== "webgpu") return;
  setStatus("setting up cislunar TLI demo…");

  const utc = "2026-01-01T00:00:00";
  let moonR: Vec3;
  try {
    const moon = await spiceState("MOON", utc, "EARTH", "J2000");
    moonR = moon.r;
  } catch {
    setStatus("SPICE kernels unavailable — run `pixi run kernels` first");
    return;
  }

  const r_leo = R_EARTH + 200e3;
  const v_leo = Math.sqrt(MU_EARTH / r_leo);
  const tof_s = 3.5 * 86400;  // 3.5 d to lunar SOI

  const lb = await optimizeLambert({
    r1_m: [r_leo, 0, 0],
    r2_m: moonR,
    tof_s,
    mu: MU_EARTH,
  });
  if (!lb.converged) {
    setStatus("Lambert diverged for cislunar setup");
    return;
  }

  // Δv1 in inertial.  At t=0 the local RIC frame is R̂=+X, Î=+Y, Ĉ=+Z so
  // ECI Δv = RIC Δv element-wise — load it directly as a RIC maneuver.
  const dv1: Vec3 = [
    lb.v1_m_s[0],
    lb.v1_m_s[1] - v_leo,
    lb.v1_m_s[2],
  ];

  editorState.body = "EARTH";
  editorState.rx = r_leo; editorState.ry = 0; editorState.rz = 0;
  editorState.vx = 0; editorState.vy = v_leo; editorState.vz = 0;
  editorState.duration_s = tof_s;
  editorState.steps = 1200;
  editorState.j2 = true; editorState.jn_max = 2;
  editorState.third_body_moon = true;
  editorState.third_body_sun = false;
  editorState.drag = false; editorState.srp = false;
  editorState.initial_mass_kg = 5000;
  editorState.maneuvers = [{
    kind: "impulsive",
    t_offset_s: 60,
    dv_r: dv1[0], dv_i: dv1[1], dv_c: dv1[2],
    duration_s: 300, thrust_n: 1000, isp_s: 300,
    dir_r: 0, dir_i: 1, dir_c: 0,
  }];
  fillIc();
  fillPerturbations();
  renderManeuverList();
  await applyEditor(renderer);

  const dvMag = Math.hypot(dv1[0], dv1[1], dv1[2]);
  const moonRangeKm = Math.hypot(moonR[0], moonR[1], moonR[2]) / 1000;
  setStatus(
    `Cislunar TLI: |Δv|=${(dvMag / 1000).toFixed(3)} km/s · ` +
    `TOF=${(tof_s / 86400).toFixed(1)} d · Moon @ ${moonRangeKm.toFixed(0)} km · ` +
    `J2 + 3rd-body Moon active`,
  );
}

/** CR3BP Earth–Moon Lagrange demo (Phase 4).
 *
 *  Fetches the five Lagrange points for the EM system, renders them as amber
 *  ✕ markers in the synodic frame, propagates a small trajectory near L1 for
 *  ~2 dimensionless units (≈9 days in EM units), and reports the worst-case
 *  Jacobi drift as a free integration-error diagnostic.  Works without SPICE
 *  — the synodic frame here is the canonical non-dimensional CR3BP frame, not
 *  the SPICE-derived one, so the demo runs on any installation. */
async function showCr3bp(renderer: Renderer): Promise<void> {
  if (renderer.kind !== "webgpu") return;
  setStatus("solving CR3BP Earth–Moon Lagrange points…");

  // Length scale = EM mean distance (m), matching backend EM_LENGTH_M.
  const L_M = 384_400_000;

  let L;
  try {
    L = await cr3bpLagrange({ system: "EARTH_MOON" });
  } catch (e) {
    setStatus(`cr3bp/lagrange error: ${(e as Error).message}`);
    return;
  }

  // Scene scale: span the EM system out to L2 + margin.
  renderer.setSceneScale(1.4 * L_M);
  // No central body — synodic frame has the barycenter at origin and two
  // primaries at known offsets; render Earth + Moon as central / secondary.
  renderer.setCentralBody(6_378_137);                    // Earth at (−μ, 0)
  // The renderer puts the central body at (0,0,0); to keep the synodic
  // convention pretty we draw both primaries explicitly.  Earth wireframe at
  // the origin is "close enough" to (−μ, 0) at our zoom; render Moon at
  // (1−μ, 0) for the secondary.
  renderer.setSecondaryBody([(1 - L.mu) * L_M - (-L.mu) * L_M, 0, 0], 1_737_400);

  // Propagate a small kick from near L1.  Choose a state that produces a
  // visible looping trajectory without escaping; the Jacobi drift over the
  // integration is reported in the status line.
  const x0 = L.L1[0] - 0.005;
  let traj;
  try {
    traj = await cr3bpPropagate({
      state: [x0, 0, 0, 0, 0.01, 0],
      t_span: [0, 6.0],
      mu: L.mu,
      steps: 1200,
    });
  } catch (e) {
    setStatus(`cr3bp/propagate error: ${(e as Error).message}`);
    return;
  }

  // Build positions buffer (scaled to SI).
  const N = traj.states.length;
  const positions = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    positions[i * 3 + 0] = traj.states[i]![0]! * L_M;
    positions[i * 3 + 1] = traj.states[i]![1]! * L_M;
    positions[i * 3 + 2] = traj.states[i]![2]! * L_M;
  }
  renderer.drawTrajectory(positions);

  // Build a synthetic states-with-velocity array so the scrubber can drive
  // the RIC marker and altitude chart.
  const statesSi: number[][] = traj.states.map(s => [
    s[0]! * L_M, s[1]! * L_M, s[2]! * L_M,
    s[3]! * L_M * 2.661699e-6,   // EM mean motion ≈ 2.66e-6 rad/s
    s[4]! * L_M * 2.661699e-6,
    s[5]! * L_M * 2.661699e-6,
  ]);
  setActiveTrajectory(traj.t.map(t => t / 2.661699e-6), statesSi);

  // Lagrange markers — set after setActiveTrajectory (which clears them).
  const toSI = (p: Vec3): Vec3 => [p[0] * L_M, p[1] * L_M, p[2] * L_M];
  renderer.setLagrangePoints([
    { position: toSI(L.L1), label: "L1" },
    { position: toSI(L.L2), label: "L2" },
    { position: toSI(L.L3), label: "L3" },
    { position: toSI(L.L4), label: "L4" },
    { position: toSI(L.L5), label: "L5" },
  ]);

  // Jacobi drift diagnostic.
  let cMin = Infinity, cMax = -Infinity;
  for (const c of traj.jacobi) {
    if (c < cMin) cMin = c;
    if (c > cMax) cMax = c;
  }
  const drift = cMax - cMin;
  setStatus(
    `CR3BP EM Lagrange: μ=${L.mu.toExponential(3)}, ` +
    `L1=${L.L1[0].toFixed(4)}, L2=${L.L2[0].toFixed(4)}, L3=${L.L3[0].toFixed(4)} · ` +
    `Jacobi drift=${drift.toExponential(2)} over ${N} samples`,
  );
}

/** Phase 4.7 — L1 Lyapunov orbit + its unstable manifold (Earth-side branch).
 *
 *  Pipeline:
 *    1. /cr3bp/periodic-orbit   → converged Lyapunov IC + period (planar L1)
 *    2. /cr3bp/propagate        → trajectory over one period (rendered orbit)
 *    3. /cr3bp/manifold         → unstable manifold − branch (interior-bound)
 *  Renders everything in the rotating EM synodic frame: Earth at origin,
 *  Moon at +d_em, periodic orbit as a closed loop around L1, manifold tubes
 *  fanning out toward Earth — the classic Conley–McGehee transit corridor. */
async function showLyapunovManifold(renderer: Renderer): Promise<void> {
  if (renderer.kind !== "webgpu") return;
  setStatus("computing L1 planar Lyapunov orbit…");

  const L_M = EM_LENGTH_M;

  // 1. Differential correction.
  let orbit;
  try {
    orbit = await cr3bpPeriodicOrbit({ family: "lyapunov", L_point: 1, Ax: 0.02 });
  } catch (e) {
    setStatus(`periodic-orbit error: ${(e as Error).message}`);
    return;
  }

  // 2. Propagate the orbit and 3. compute the manifold concurrently.
  setStatus(`L1 Lyapunov: T=${orbit.period.toFixed(3)}, C=${orbit.jacobi.toFixed(4)} — computing manifold…`);
  let propResp, manResp;
  try {
    [propResp, manResp] = await Promise.all([
      cr3bpPropagate({
        state: orbit.state0,
        t_span: [0, orbit.period],
        mu: orbit.mu,
        steps: 600,
      }),
      cr3bpManifold({
        orbit_state: orbit.state0,
        period: orbit.period,
        mu: orbit.mu,
        direction: "unstable",
        branch: "-",       // − branch points back toward Earth from L1
        n_samples: 30,
        duration: 4.0,
        steps: 200,
      }),
    ]);
  } catch (e) {
    setStatus(`manifold error: ${(e as Error).message}`);
    return;
  }

  // Scene scale spans the L1 → Moon corridor with a margin.
  renderer.setSceneScale(1.4 * L_M);
  renderer.setCentralBody(BODY.EARTH.radius);

  // Earth-centric synodic view: barycentric x + μ·d_em shift.
  const xShift = EM_MU_FRONTEND * L_M;
  const orbitPositions = new Float32Array(propResp.states.length * 3);
  for (let i = 0; i < propResp.states.length; i++) {
    orbitPositions[i * 3 + 0] = propResp.states[i]![0]! * L_M + xShift;
    orbitPositions[i * 3 + 1] = propResp.states[i]![1]! * L_M;
    orbitPositions[i * 3 + 2] = propResp.states[i]![2]! * L_M;
  }
  renderer.drawTrajectory(orbitPositions);

  // Build manifold-tube positions in the same Earth-centric synodic frame.
  const tubes: Float32Array<ArrayBuffer>[] = [];
  for (const tube of manResp.trajectories) {
    if (tube.length < 2) continue;
    const arr = new Float32Array(tube.length * 3);
    for (let i = 0; i < tube.length; i++) {
      arr[i * 3 + 0] = tube[i]![0]! * L_M + xShift;
      arr[i * 3 + 1] = tube[i]![1]! * L_M;
      arr[i * 3 + 2] = tube[i]![2]! * L_M;
    }
    tubes.push(arr);
  }
  renderer.setManifoldTubes(tubes);

  // Static landmarks in the synodic frame.
  renderer.setSecondaryBody([L_M, 0, 0], BODY.MOON.radius);
  latestMoonR = [L_M, 0, 0];
  moonTrackT = null; moonTrackR = null;

  // Lagrange-point ✕ markers at synodic positions (Earth-centric shift).
  const L = await cr3bpLagrange({ system: "EARTH_MOON" });
  const toSI = (p: [number, number, number]): [number, number, number] =>
    [p[0] * L_M + xShift, p[1] * L_M, p[2] * L_M];
  // (defer setLagrangePoints until after setActiveTrajectory clears them)

  // Scrubber needs a synthetic states-with-velocity for the orbit.
  const n = 2.661699e-6;
  const statesSi: number[][] = propResp.states.map(s => [
    s[0]! * L_M + xShift, s[1]! * L_M, s[2]! * L_M,
    s[3]! * L_M * n, s[4]! * L_M * n, s[5]! * L_M * n,
  ]);
  setActiveTrajectory(propResp.t.map(t => t / n), statesSi);

  renderer.setLagrangePoints([
    { position: toSI(L.L1), label: "L1" },
    { position: toSI(L.L2), label: "L2" },
    { position: toSI(L.L3), label: "L3" },
    { position: toSI(L.L4), label: "L4" },
    { position: toSI(L.L5), label: "L5" },
  ]);
  refreshCameraTarget();

  setStatus(
    `L1 Lyapunov + unstable manifold (− branch): ` +
    `Ax=0.02 nd · T=${orbit.period.toFixed(3)} · C=${orbit.jacobi.toFixed(4)} · ` +
    `${tubes.length}/${manResp.trajectories.length} tubes rendered · DC ${orbit.dc_iterations} iter`,
  );
}


/** Phase 4.8 — Weak Stability Boundary diagnostic.
 *
 *  Sweeps a grid of (altitude, angle) initial conditions in low lunar orbit,
 *  propagates each backward in time, and classifies as captured (stayed near
 *  the Moon for the integration window) or escaped (came in from infinity).
 *  Renders captured ICs as green dots and escapes as faint red dots, plotted
 *  in the EM synodic frame so the Moon is stationary on +x. */
async function showWsb(renderer: Renderer): Promise<void> {
  if (renderer.kind !== "webgpu") return;
  setStatus("computing WSB capture/escape grid…");

  // Coarse grid: 6 altitudes × 12 angles = 72 propagations.  Tight enough
  // to see the WSB filament without overloading the backend.
  const altitudes_m = [50e3, 200e3, 500e3, 1_000e3, 2_000e3, 5_000e3];
  const angles_rad: number[] = [];
  for (let i = 0; i < 12; i++) angles_rad.push((i / 12) * 2 * Math.PI);

  let res;
  try {
    res = await cr3bpWsb({ altitudes_m, angles_rad, duration: 4.0 });
  } catch (e) {
    setStatus(`wsb error: ${(e as Error).message}`);
    return;
  }

  // Scene set-up: Earth-centric synodic, Moon as the static landmark.
  const L_M = EM_LENGTH_M;
  const xShift = EM_MU_FRONTEND * L_M;
  renderer.setSceneScale(1.4 * L_M);
  renderer.setCentralBody(BODY.EARTH.radius);
  renderer.setSecondaryBody([L_M, 0, 0], BODY.MOON.radius);
  latestMoonR = [L_M, 0, 0];
  moonTrackT = null; moonTrackR = null;

  // Encode each grid cell as a tiny "trajectory" of two points (a short
  // segment) so the existing manifold-tubes infrastructure can render the
  // diagnostic in one upload.
  const captured: Float32Array<ArrayBuffer>[] = [];
  const escaped: Float32Array<ArrayBuffer>[] = [];
  const moonX = (1.0 - EM_MU_FRONTEND) * L_M + xShift;  // Moon in Earth-centric synodic
  for (let i = 0; i < altitudes_m.length; i++) {
    const r_m = BODY.MOON.radius + altitudes_m[i]!;
    for (let j = 0; j < angles_rad.length; j++) {
      const theta = angles_rad[j]!;
      const px = moonX + r_m * Math.cos(theta);
      const py = r_m * Math.sin(theta);
      // Tiny radial spur so the renderer has a non-degenerate segment.
      const spur = 0.0035 * L_M;
      const seg = new Float32Array([
        px, py, 0,
        px + spur * Math.cos(theta), py + spur * Math.sin(theta), 0,
      ]);
      const cls = res.grid[i]![j]!;
      if (cls === 1) captured.push(seg);
      else if (cls === 0) escaped.push(seg);
    }
  }
  // Two-pass render: escaped (dim) first via main trajectory buffer, then
  // captured (bright violet) via the manifold-tubes pass on top.
  // For simplicity reuse the manifold-tubes infrastructure with a colour
  // override per call.
  renderer.setManifoldTubes(captured, [0.30, 1.00, 0.40]);   // bright green
  // Plot escapes as a single "trajectory" by concatenating segments — they're
  // already short, so the polyline interpretation just draws each in sequence.
  // (We could split them up, but for a coarse diagnostic the visual is fine.)
  if (escaped.length > 0) {
    let total = 0;
    for (const e of escaped) total += e.length;
    const merged = new Float32Array(total);
    let o = 0;
    for (const e of escaped) { merged.set(e, o); o += e.length; }
    renderer.drawTrajectory(merged);
  } else {
    // No escapes: draw the periapsis circle around the Moon as a context cue.
    const ring = new Float32Array(64 * 3);
    for (let k = 0; k < 64; k++) {
      const t = (k / 63) * 2 * Math.PI;
      const r = BODY.MOON.radius + altitudes_m[0]!;
      ring[k * 3] = moonX + r * Math.cos(t);
      ring[k * 3 + 1] = r * Math.sin(t);
      ring[k * 3 + 2] = 0;
    }
    renderer.drawTrajectory(ring);
  }

  // Synthetic single-state "trajectory" so the scrubber doesn't complain.
  setActiveTrajectory([0], [[moonX, 0, 0, 0, 0, 0]]);

  // Lagrange-point ✕ markers at synodic positions for context.
  const L = await cr3bpLagrange({ system: "EARTH_MOON" });
  const toSI = (p: [number, number, number]): [number, number, number] =>
    [p[0] * L_M + xShift, p[1] * L_M, p[2] * L_M];
  renderer.setLagrangePoints([
    { position: toSI(L.L1), label: "L1" },
    { position: toSI(L.L2), label: "L2" },
  ]);
  refreshCameraTarget();

  const totalCells = altitudes_m.length * angles_rad.length;
  setStatus(
    `WSB diagnostic: ${captured.length}/${totalCells} captured · ` +
    `${escaped.length}/${totalCells} escaped · ` +
    `${totalCells - captured.length - escaped.length} failed/uncertain`,
  );
}


// --------------------------------------------------------------------------- //
//  Maneuver editor — sidebar panel that lets the user set IC, add Δv kicks,
//  and re-propagate the trajectory.
// --------------------------------------------------------------------------- //

type ManeuverRow = {
  kind: "impulsive" | "finite";
  t_offset_s: number;   // impulse time OR burn start time
  // impulsive fields
  dv_r: number; dv_i: number; dv_c: number;
  // finite burn fields
  duration_s: number;
  thrust_n: number;
  isp_s: number;
  dir_r: number; dir_i: number; dir_c: number;
};

type ViewFrame = "J2000" | "EM_SYNODIC";

const editorState: {
  body: BodyName;
  frame: ViewFrame;
  rx: number; ry: number; rz: number;
  vx: number; vy: number; vz: number;
  duration_s: number;
  steps: number;
  initial_mass_kg: number;
  maneuvers: ManeuverRow[];
  j2: boolean;
  jn_max: number;
  drag: boolean;
  drag_model: "exponential" | "msis";
  drag_mass_kg: number;
  drag_area_m2: number;
  drag_cd: number;
  srp: boolean;
  srp_area_m2: number;
  srp_cr: number;
  third_body_moon: boolean;
  third_body_sun: boolean;
} = {
  body: "EARTH",
  frame: "J2000",
  rx: R0, ry: 0, rz: 0,
  vx: 0, vy: Math.sqrt(MU_EARTH / R0), vz: 0,
  duration_s: 2 * Math.PI * Math.sqrt((R0 * R0 * R0) / MU_EARTH),
  steps: 400,
  initial_mass_kg: 1000,
  maneuvers: [],
  j2: false,
  jn_max: 2,
  drag: false,
  drag_model: "exponential",
  drag_mass_kg: 500,
  drag_area_m2: 4.0,
  drag_cd: 2.2,
  srp: false,
  srp_area_m2: 4.0,
  srp_cr: 1.5,
  third_body_moon: false,
  third_body_sun: false,
};

function fillPerturbations(): void {
  const set = (id: string, v: string | boolean) => {
    const el = document.getElementById(id) as HTMLInputElement | HTMLSelectElement;
    if (typeof v === "boolean") (el as HTMLInputElement).checked = v;
    else el.value = v;
  };
  set("pert-j2", editorState.j2);
  set("pert-jn-max", String(editorState.jn_max));
  set("pert-drag", editorState.drag);
  set("pert-drag-model", editorState.drag_model);
  set("pert-srp", editorState.srp);
  set("pert-3b-moon", editorState.third_body_moon);
  set("pert-3b-sun", editorState.third_body_sun);
  set("pert-mass", String(editorState.drag_mass_kg));
  set("pert-area", String(editorState.drag_area_m2));
  set("pert-cd", String(editorState.drag_cd));
  set("pert-srp-area", String(editorState.srp_area_m2));
  set("pert-srp-cr", String(editorState.srp_cr));
  updatePertVehicleVisibility();
}

function updatePertVehicleVisibility(): void {
  const needVehicle = editorState.drag || editorState.srp;
  (document.getElementById("pert-vehicle") as HTMLElement).style.display =
    needVehicle ? "block" : "none";
}

function readPerturbations(): void {
  const chk = (id: string) => (document.getElementById(id) as HTMLInputElement).checked;
  const n = (id: string) => parseFloat((document.getElementById(id) as HTMLInputElement).value);
  const s = (id: string) => (document.getElementById(id) as HTMLSelectElement).value;
  editorState.j2 = chk("pert-j2");
  editorState.jn_max = parseInt(s("pert-jn-max"), 10);
  editorState.drag = chk("pert-drag");
  editorState.drag_model = s("pert-drag-model") as "exponential" | "msis";
  editorState.srp = chk("pert-srp");
  editorState.third_body_moon = chk("pert-3b-moon");
  editorState.third_body_sun = chk("pert-3b-sun");
  editorState.drag_mass_kg = n("pert-mass");
  editorState.drag_area_m2 = n("pert-area");
  editorState.drag_cd = n("pert-cd");
  editorState.srp_area_m2 = n("pert-srp-area");
  editorState.srp_cr = n("pert-srp-cr");
}

function fillIc(): void {
  (document.getElementById("ic-body") as HTMLSelectElement).value = editorState.body;
  (document.getElementById("ic-frame") as HTMLSelectElement).value = editorState.frame;
  (document.getElementById("ic-rx") as HTMLInputElement).value = String(editorState.rx);
  (document.getElementById("ic-ry") as HTMLInputElement).value = String(editorState.ry);
  (document.getElementById("ic-rz") as HTMLInputElement).value = String(editorState.rz);
  (document.getElementById("ic-vx") as HTMLInputElement).value = String(editorState.vx);
  (document.getElementById("ic-vy") as HTMLInputElement).value = String(editorState.vy);
  (document.getElementById("ic-vz") as HTMLInputElement).value = String(editorState.vz);
  (document.getElementById("ic-duration") as HTMLInputElement).value = editorState.duration_s.toFixed(0);
  (document.getElementById("ic-steps") as HTMLInputElement).value = String(editorState.steps);
  (document.getElementById("ic-mass") as HTMLInputElement).value = String(editorState.initial_mass_kg);
}

function readIc(): void {
  const n = (id: string) => parseFloat((document.getElementById(id) as HTMLInputElement).value);
  editorState.body = (document.getElementById("ic-body") as HTMLSelectElement).value as BodyName;
  editorState.frame = (document.getElementById("ic-frame") as HTMLSelectElement).value as ViewFrame;
  editorState.rx = n("ic-rx"); editorState.ry = n("ic-ry"); editorState.rz = n("ic-rz");
  editorState.vx = n("ic-vx"); editorState.vy = n("ic-vy"); editorState.vz = n("ic-vz");
  editorState.duration_s = n("ic-duration");
  editorState.steps = parseInt((document.getElementById("ic-steps") as HTMLInputElement).value, 10);
  editorState.initial_mass_kg = n("ic-mass");
}

/** Repopulate the IC fields with a circular equatorial orbit at the body's
 *  default altitude.  Called when the user changes the central-body selector. */
function applyBodyDefaults(body: BodyName): void {
  const b = BODY[body];
  const r0 = b.radius + b.default_alt_m;
  const v0 = Math.sqrt(b.mu / r0);
  editorState.body = body;
  editorState.rx = r0; editorState.ry = 0; editorState.rz = 0;
  editorState.vx = 0;  editorState.vy = v0; editorState.vz = 0;
  editorState.duration_s = 2 * Math.PI * Math.sqrt((r0 * r0 * r0) / b.mu);
  // Drag/SRP atmospheric envelope only makes sense for Earth — clear them
  // so the user isn't surprised by a force that the backend will reject.
  if (body !== "EARTH") {
    editorState.drag = false; editorState.srp = false;
  }
  fillIc();
  fillPerturbations();
}

function renderManeuverList(): void {
  const list = document.getElementById("man-list")!;
  list.innerHTML = "";
  if (editorState.maneuvers.length === 0) {
    list.innerHTML = '<div style="opacity:.6">no maneuvers — click "+ Add" to insert one</div>';
    return;
  }
  editorState.maneuvers.forEach((m, idx) => {
    const fin = m.kind === "finite";
    const div = document.createElement("div");
    div.className = "maneuver";
    div.innerHTML = `
      <div class="row" style="margin-bottom:3px">
        <strong>#${idx + 1}</strong>
        <select data-i="${idx}" data-k="kind" style="font:11px inherit;background:#020a02;color:#80ff60;border:1px solid #2a602a">
          <option value="impulsive"${!fin ? " selected" : ""}>Impulsive Δv</option>
          <option value="finite"${fin ? " selected" : ""}>Finite burn</option>
        </select>
        <label style="flex:1">t${fin ? "_start" : ""}&nbsp;(s)
          <input data-i="${idx}" data-k="t_offset_s" type="number" step="60" value="${m.t_offset_s}" />
        </label>
        <button data-rm="${idx}" class="danger">×</button>
      </div>
      ${!fin ? `
      <div class="vec3">
        <input data-i="${idx}" data-k="dv_r" type="number" step="10" value="${m.dv_r}" placeholder="ΔvR (m/s)" />
        <input data-i="${idx}" data-k="dv_i" type="number" step="10" value="${m.dv_i}" placeholder="ΔvI (m/s)" />
        <input data-i="${idx}" data-k="dv_c" type="number" step="10" value="${m.dv_c}" placeholder="ΔvC (m/s)" />
      </div>` : `
      <div class="row" style="gap:3px;flex-wrap:wrap">
        <label style="flex:1">dur&nbsp;(s)<input data-i="${idx}" data-k="duration_s" type="number" step="10" min="1" value="${m.duration_s}" /></label>
        <label style="flex:1">T&nbsp;(N)<input data-i="${idx}" data-k="thrust_n" type="number" step="100" min="0.001" value="${m.thrust_n}" /></label>
        <label style="flex:1">Isp&nbsp;(s)<input data-i="${idx}" data-k="isp_s" type="number" step="10" min="1" value="${m.isp_s}" /></label>
      </div>
      <div class="vec3" style="margin-top:2px">
        <input data-i="${idx}" data-k="dir_r" type="number" step="0.1" value="${m.dir_r}" placeholder="dirR" />
        <input data-i="${idx}" data-k="dir_i" type="number" step="0.1" value="${m.dir_i}" placeholder="dirI" />
        <input data-i="${idx}" data-k="dir_c" type="number" step="0.1" value="${m.dir_c}" placeholder="dirC" />
      </div>`}
    `;
    list.appendChild(div);
  });

  // Kind selector — re-render the row on change.
  list.querySelectorAll("select[data-i]").forEach((el) => {
    el.addEventListener("change", () => {
      const i = parseInt(el.getAttribute("data-i")!, 10);
      editorState.maneuvers[i]!.kind = (el as HTMLSelectElement).value as "impulsive" | "finite";
      renderManeuverList();
    });
  });

  // Numeric inputs — write directly using a string-keyed cast to avoid type noise.
  list.querySelectorAll("input[data-i]").forEach((el) => {
    el.addEventListener("input", () => {
      const i = parseInt(el.getAttribute("data-i")!, 10);
      const k = el.getAttribute("data-k")!;
      const v = parseFloat((el as HTMLInputElement).value);
      if (Number.isFinite(v)) (editorState.maneuvers[i]! as Record<string, unknown>)[k] = v;
    });
  });

  list.querySelectorAll("button[data-rm]").forEach((el) => {
    el.addEventListener("click", () => {
      const i = parseInt(el.getAttribute("data-rm")!, 10);
      editorState.maneuvers.splice(i, 1);
      renderManeuverList();
    });
  });
}

async function applyEditor(renderer: Renderer): Promise<void> {
  readIc();
  readPerturbations();
  setStatus("propagating editor scenario…");
  const b = BODY[editorState.body];
  const data = await propagate({
    state: {
      r: [editorState.rx, editorState.ry, editorState.rz],
      v: [editorState.vx, editorState.vy, editorState.vz],
    },
    duration_s: editorState.duration_s,
    steps: editorState.steps,
    body_name: editorState.body,
    mu: b.mu,
    body_radius: b.radius,
    ...(editorState.j2 ? { j2_enabled: true as const, jn_max: editorState.jn_max } : {}),
    ...((editorState.drag || editorState.srp) ? {
      vehicle: {
        mass_kg: editorState.drag_mass_kg,
        drag_area_m2: editorState.drag_area_m2,
        drag_cd: editorState.drag_cd,
        srp_area_m2: editorState.srp_area_m2,
        srp_cr: editorState.srp_cr,
      },
    } : {}),
    ...(editorState.drag ? { drag: true as const, drag_model: editorState.drag_model } : {}),
    ...(editorState.srp ? { srp: true as const } : {}),
    ...(((editorState.third_body_moon || editorState.third_body_sun) ? {
      third_body: [
        ...(editorState.third_body_moon ? ["MOON"] : []),
        ...(editorState.third_body_sun ? ["SUN"] : []),
      ],
    } : {}) as { third_body?: string[] }),
    maneuvers: editorState.maneuvers
      .filter((m) => m.kind === "impulsive")
      .map((m) => ({ t_offset_s: m.t_offset_s, dv_ric: [m.dv_r, m.dv_i, m.dv_c] as [number, number, number] })),
    ...(editorState.maneuvers.some((m) => m.kind === "finite") ? {
      finite_burns: editorState.maneuvers
        .filter((m): m is ManeuverRow & { kind: "finite" } => m.kind === "finite")
        .map((m): FiniteBurnSpec => ({
          t_start_s: m.t_offset_s,
          duration_s: m.duration_s,
          thrust_n: m.thrust_n,
          isp_s: m.isp_s,
          direction_ric: [m.dir_r, m.dir_i, m.dir_c],
        })),
      initial_mass_kg: editorState.initial_mass_kg,
    } : {}),
  });

  // Frame post-processing.  When the user selects EM_SYNODIC, transform the
  // trajectory states through the backend so the Earth--Moon line is along
  // +x and both primaries appear stationary.  This requires SPICE kernels;
  // fall back silently to inertial on error.
  const wantSynodic = editorState.frame === "EM_SYNODIC" && editorState.body === "EARTH";
  let renderedStates = data.states;
  let synodicActive = false;
  if (wantSynodic) {
    try {
      const tr = await transformStates({
        direction: "to_synodic",
        t_tdb: 0.0,  // J2000-relative reference; the per-state offsets carry the rest
        t_offsets_s: data.t,
        states: data.states,
      });
      // Backend returns barycentric synodic; shift origin to Earth-center so
      // the central-body wireframe at (0,0,0) lines up.
      const xShift = EM_MU_FRONTEND * EM_LENGTH_M;
      renderedStates = tr.states.map((s) => [
        s[0]! + xShift, s[1]!, s[2]!, s[3]!, s[4]!, s[5]!,
      ]);
      synodicActive = true;
    } catch {
      setStatus("EM_SYNODIC requires SPICE kernels — falling back to J2000");
    }
  }

  // Auto-scale the scene to fit the trajectory (max radius * 1.2).
  let maxR = 0;
  for (const s of renderedStates) {
    const r = Math.hypot(s[0]!, s[1]!, s[2]!);
    if (r > maxR) maxR = r;
  }
  // Scene scale: include the Moon range when it'll be visible (third-body
  // Moon in J2000, or always in synodic where the Moon is a static landmark).
  const willRenderMoon = synodicActive
    || (editorState.body === "EARTH" && editorState.third_body_moon);
  const moonRange = 4.0e8;  // approx Earth-Moon distance, used as a min scale
  const sceneScale = willRenderMoon
    ? Math.max(b.radius * 1.1, maxR * 1.2, moonRange * 1.2)
    : Math.max(b.radius * 1.1, maxR * 1.2);
  renderer.setSceneScale(sceneScale);
  renderer.setCentralBody(b.radius);

  // Moon rendering branch.
  if (synodicActive) {
    // Static landmark in the rotating frame.  Moon sits at +d_em from Earth-
    // center (= +(1−μ)·d_em from barycenter + μ·d_em shift).
    latestMoonR = [EM_LENGTH_M, 0, 0];
    moonTrackT = null; moonTrackR = null;
    renderer.setSecondaryBody(latestMoonR, BODY.MOON.radius);
  } else if (willRenderMoon) {
    try {
      // Sample Moon ephemeris at ~60 points across the trajectory so the
      // scrubber-driven update is smooth without overloading SPICE.
      const N = data.t.length;
      const target_samples = Math.min(60, N);
      const stride = Math.max(1, Math.floor(N / target_samples));
      const offsets: number[] = [];
      for (let i = 0; i < N; i += stride) offsets.push(data.t[i]!);
      if (offsets[offsets.length - 1] !== data.t[N - 1]) offsets.push(data.t[N - 1]!);

      const eph = await spiceEphemeris("MOON", "2026-01-01T00:00:00", offsets, "EARTH", "J2000");
      moonTrackT = offsets;
      moonTrackR = eph.r;
      latestMoonR = eph.r[0]!;
      renderer.setSecondaryBody(latestMoonR, BODY.MOON.radius);
    } catch {
      moonTrackT = null; moonTrackR = null; latestMoonR = null;
      renderer.setSecondaryBody(null, 0);
    }
  } else {
    moonTrackT = null; moonTrackR = null; latestMoonR = null;
    renderer.setSecondaryBody(null, 0);
  }
  refreshCameraTarget();

  // Phase colours by manoeuvre boundaries (all events, sorted).
  const phaseTimes = [0, ...editorState.maneuvers.map((m) => m.t_offset_s).sort((a, b) => a - b)];
  const phaseColors = phaseTimes.map((_, i) => {
    if (i === 0) return C_DEPART;
    if (i === phaseTimes.length - 1) return C_ARRIVE;
    return C_TRANSFER;
  });
  const burnWindows = editorState.maneuvers
    .filter((m) => m.kind === "finite")
    .map((m): [number, number] => [m.t_offset_s, m.t_offset_s + m.duration_s]);
  const colors = phaseColors.length > 1
    ? applyBurnWindows(colorByPhases(data.t, phaseTimes, phaseColors), data.t, burnWindows)
    : burnWindows.length > 0
      ? applyBurnWindows(
          colorByPhases(data.t, [0], [C_DEPART]),
          data.t,
          burnWindows,
        )
      : undefined;
  renderer.drawTrajectory(statesToPositions(renderedStates), colors);
  setActiveTrajectory(data.t, renderedStates);

  // Active-perturbations badge — what the server actually applied.
  const badge = document.getElementById("pert-active");
  if (badge) {
    badge.textContent = data.perturbations && data.perturbations.length > 0
      ? "active: " + data.perturbations.join(" · ")
      : "no perturbations (two-body)";
  }

  // Summary.
  const totalImpDv = editorState.maneuvers
    .filter((m) => m.kind === "impulsive")
    .reduce((s, m) => s + Math.hypot(m.dv_r, m.dv_i, m.dv_c), 0);
  const finCount = editorState.maneuvers.filter((m) => m.kind === "finite").length;
  const summary = document.getElementById("editor-summary")!;
  summary.innerHTML =
    `samples: ${data.states.length}<br>` +
    `peak |r|: ${(maxR / 1000).toFixed(0)} km<br>` +
    (totalImpDv > 0 ? `Σ imp. |Δv|: ${(totalImpDv / 1000).toFixed(3)} km/s<br>` : "") +
    (finCount > 0 ? `finite burns: ${finCount}` : "");
  setStatus(
    `Editor: ${editorState.maneuvers.length} event(s)` +
    (totalImpDv > 0 ? `, Σ imp. Δv=${(totalImpDv / 1000).toFixed(2)} km/s` : "") +
    (finCount > 0 ? `, ${finCount} finite burn(s)` : ""),
  );
}

function resetEditor(): void {
  editorState.body = "EARTH";
  editorState.frame = "J2000";
  editorState.steps = 400;
  editorState.maneuvers = [];
  editorState.initial_mass_kg = 1000;
  editorState.j2 = false; editorState.jn_max = 2;
  editorState.drag = false; editorState.drag_model = "exponential";
  editorState.drag_mass_kg = 500; editorState.drag_area_m2 = 4.0; editorState.drag_cd = 2.2;
  editorState.srp = false; editorState.srp_area_m2 = 4.0; editorState.srp_cr = 1.5;
  editorState.third_body_moon = false; editorState.third_body_sun = false;
  applyBodyDefaults("EARTH");
  renderManeuverList();
}

// --------------------------------------------------------------------------- //
//  Solver panel — Hohmann, Lambert, Multi-burn NLP, TLE.
// --------------------------------------------------------------------------- //

function makeImpulse(t: number, dv: Vec3): ManeuverRow {
  return {
    kind: "impulsive",
    t_offset_s: t,
    dv_r: dv[0], dv_i: dv[1], dv_c: dv[2],
    duration_s: 300, thrust_n: 1000, isp_s: 300,
    dir_r: 0, dir_i: 1, dir_c: 0,
  };
}

function renderSolverForm(): void {
  const type = (document.getElementById("solver-type") as HTMLSelectElement).value;
  const form = document.getElementById("solver-form")!;
  const b = BODY[editorState.body];
  const r1Default = b.radius + b.default_alt_m;
  const r2Default = editorState.body === "EARTH" ? 42_164_000 : b.radius + 1_000_000;

  if (type === "hohmann") {
    form.innerHTML = `
      <label>r₁ (m)<input id="sv-r1" type="number" value="${r1Default}" step="100000" /></label>
      <label>r₂ (m)<input id="sv-r2" type="number" value="${r2Default}" step="100000" /></label>`;
  } else if (type === "lambert") {
    form.innerHTML = `
      <div style="opacity:.7;margin-bottom:2px">r⃗₁ (m)</div>
      <div class="vec3">
        <input id="sv-r1x" type="number" value="${r1Default}" />
        <input id="sv-r1y" type="number" value="0" />
        <input id="sv-r1z" type="number" value="0" />
      </div>
      <div style="opacity:.7;margin:3px 0 2px">r⃗₂ (m)</div>
      <div class="vec3">
        <input id="sv-r2x" type="number" value="${(-r2Default * 0.5).toFixed(0)}" />
        <input id="sv-r2y" type="number" value="${(r2Default * 0.87).toFixed(0)}" />
        <input id="sv-r2z" type="number" value="0" />
      </div>
      <label>TOF (s)<input id="sv-tof" type="number" value="14400" step="60" /></label>`;
  } else if (type === "multi-burn") {
    form.innerHTML = `
      <div style="opacity:.7;margin-bottom:2px">Initial state taken from editor IC.</div>
      <div style="opacity:.7;margin-bottom:2px">Target r⃗_f (m)</div>
      <div class="vec3">
        <input id="sv-xf-rx" type="number" value="${r2Default}" />
        <input id="sv-xf-ry" type="number" value="0" />
        <input id="sv-xf-rz" type="number" value="0" />
      </div>
      <div style="opacity:.7;margin:3px 0 2px">Target v⃗_f (m/s)</div>
      <div class="vec3">
        <input id="sv-xf-vx" type="number" value="0" />
        <input id="sv-xf-vy" type="number" value="${Math.sqrt(b.mu / r2Default).toFixed(2)}" />
        <input id="sv-xf-vz" type="number" value="0" />
      </div>
      <label>Burn epochs (s, comma-sep)
        <input id="sv-epochs" type="text" value="600, 18000" /></label>
      <label>t_final (s)<input id="sv-tfin" type="number" value="36000" step="600" /></label>`;
  } else if (type === "tle") {
    form.innerHTML = `
      <label>NORAD ID (fetch from Celestrak)
        <input id="sv-norad" type="number" placeholder="25544" step="1" /></label>
      <div style="opacity:.7;margin:3px 0 2px">— or paste TLE —</div>
      <input id="sv-l1" type="text" placeholder="Line 1" style="width:100%;background:#020a02;color:#80ff60;border:1px solid #2a602a;font:10px ui-monospace,monospace;padding:2px" />
      <input id="sv-l2" type="text" placeholder="Line 2" style="width:100%;background:#020a02;color:#80ff60;border:1px solid #2a602a;font:10px ui-monospace,monospace;padding:2px;margin-top:2px" />
      <label>at_utc (optional ISO-8601)
        <input id="sv-utc" type="text" placeholder="2026-01-01T00:00:00" /></label>`;
  }
}

async function runSolver(renderer: Renderer): Promise<void> {
  const type = (document.getElementById("solver-type") as HTMLSelectElement).value;
  const info = document.getElementById("solver-info")!;
  const b = BODY[editorState.body];
  const n = (id: string) => parseFloat((document.getElementById(id) as HTMLInputElement).value);
  const fmt = (x: number) => (x / 1000).toFixed(3);

  try {
    if (type === "hohmann") {
      const r1 = n("sv-r1"), r2 = n("sv-r2");
      const res = await optimizeHohmann({ r1_m: r1, r2_m: r2, mu: b.mu });
      const v1 = Math.sqrt(b.mu / r1);
      editorState.rx = r1; editorState.ry = 0; editorState.rz = 0;
      editorState.vx = 0;  editorState.vy = v1; editorState.vz = 0;
      editorState.duration_s = res.transfer_time_s + 600;
      editorState.maneuvers = [
        makeImpulse(60, [0, res.dv1_m_s, 0]),
        makeImpulse(60 + res.transfer_time_s, [0, res.dv2_m_s, 0]),
      ];
      info.innerHTML =
        `Δv₁=${fmt(res.dv1_m_s)} km/s · Δv₂=${fmt(res.dv2_m_s)} km/s<br>` +
        `total=${fmt(res.dv_total_m_s)} km/s · TOF=${(res.transfer_time_s / 3600).toFixed(2)} h<br>` +
        `sma=${(res.semi_major_axis_m / 1000).toFixed(0)} km`;
    } else if (type === "lambert") {
      const r1: Vec3 = [n("sv-r1x"), n("sv-r1y"), n("sv-r1z")];
      const r2: Vec3 = [n("sv-r2x"), n("sv-r2y"), n("sv-r2z")];
      const tof = n("sv-tof");
      const res = await optimizeLambert({ r1_m: r1, r2_m: r2, tof_s: tof, mu: b.mu });
      editorState.rx = r1[0]; editorState.ry = r1[1]; editorState.rz = r1[2];
      editorState.vx = res.v1_m_s[0]; editorState.vy = res.v1_m_s[1]; editorState.vz = res.v1_m_s[2];
      editorState.duration_s = tof;
      editorState.maneuvers = [];
      const v1n = Math.hypot(...res.v1_m_s), v2n = Math.hypot(...res.v2_m_s);
      info.innerHTML =
        `${res.converged ? "✓ converged" : "✗ diverged"} (${res.iterations} iter)<br>` +
        `|v⃗₁|=${fmt(v1n)} km/s · |v⃗₂|=${fmt(v2n)} km/s<br>` +
        `TOF=${(tof / 3600).toFixed(2)} h`;
    } else if (type === "multi-burn") {
      readIc();
      const epochs = (document.getElementById("sv-epochs") as HTMLInputElement).value
        .split(",").map(s => parseFloat(s.trim())).filter(Number.isFinite);
      const tfin = n("sv-tfin");
      const res = await optimizeMultiBurn({
        x0_r: [editorState.rx, editorState.ry, editorState.rz],
        x0_v: [editorState.vx, editorState.vy, editorState.vz],
        xf_r: [n("sv-xf-rx"), n("sv-xf-ry"), n("sv-xf-rz")],
        xf_v: [n("sv-xf-vx"), n("sv-xf-vy"), n("sv-xf-vz")],
        maneuver_epochs_s: epochs,
        t_final_s: tfin,
        mu: b.mu,
      });
      editorState.duration_s = tfin;
      // NLP returns inertial Δvs.  Our editor speaks RIC, so we can't 1:1
      // round-trip without propagating to each epoch; print the result and
      // let the user adjust.
      info.innerHTML =
        `${res.converged ? "✓ converged" : "✗ diverged"} (${res.iterations} iter)<br>` +
        `Σ |Δv| = ${fmt(res.total_dv_m_s)} km/s<br>` +
        res.dv_inertial_m_s.map((dv, i) => {
          const mag = Math.hypot(...dv);
          return `Δv${i + 1}@${epochs[i]}s: |${fmt(mag)}| km/s (inertial)`;
        }).join("<br>") +
        `<br><span style="opacity:.6">(inertial Δv — not auto-loaded as RIC)</span>`;
    } else if (type === "tle") {
      const noradStr = (document.getElementById("sv-norad") as HTMLInputElement).value;
      const utc = (document.getElementById("sv-utc") as HTMLInputElement).value || undefined;
      const norad = parseFloat(noradStr);
      const resp = Number.isFinite(norad)
        ? await tleByNorad(norad, utc)
        : await tleParse(
            (document.getElementById("sv-l1") as HTMLInputElement).value,
            (document.getElementById("sv-l2") as HTMLInputElement).value,
            "",
            utc,
          );
      const r = resp.state.r, v = resp.state.v;
      const rmag = Math.hypot(r[0], r[1], r[2]);
      const period = 2 * Math.PI * Math.sqrt(rmag ** 3 / b.mu);
      // TLEs are Earth-only; force the body selector to EARTH for correctness.
      editorState.body = "EARTH";
      editorState.rx = r[0]; editorState.ry = r[1]; editorState.rz = r[2];
      editorState.vx = v[0]; editorState.vy = v[1]; editorState.vz = v[2];
      editorState.duration_s = period;
      editorState.maneuvers = [];
      info.innerHTML =
        `<strong>${resp.name || "(unnamed)"}</strong> NORAD ${resp.norad_id}<br>` +
        `alt=${resp.altitude_km.toFixed(0)} km · |v|=${fmt(resp.speed_m_s)} km/s<br>` +
        `period=${(period / 60).toFixed(1)} min`;
    }
    fillIc();
    renderManeuverList();
    await applyEditor(renderer);
  } catch (e) {
    info.textContent = `error: ${(e as Error).message}`;
    setStatus(`solver failed: ${(e as Error).message}`);
  }
}

async function main(): Promise<void> {
  const renderer = await initRenderer(canvas);
  activeRenderer = renderer;
  if (renderer.kind !== "webgpu") {
    showBanner(
      "WebGPU is not available in this browser. " +
      "Try Chrome 113+, Edge 113+, or Safari 18.2+ on a supported GPU. " +
      "Firefox needs `dom.webgpu.enabled` flipped in about:config.",
    );
    setStatus("renderer disabled — backend calls still work via /docs");
    return;
  }

  try {
    const h = await fetchHealth();
    version.textContent = `v${h.version}`;
  } catch (e) {
    setStatus(`backend unreachable: ${(e as Error).message}`);
    return;
  }

  const wire = (
    btn: HTMLButtonElement,
    fn: () => Promise<void>,
  ): void => {
    btn.addEventListener("click", () => {
      selectButton(btn);
      fn().catch((e) => setStatus(`error: ${(e as Error).message}`));
    });
  };
  wire(btnLeo,     () => showLeo(renderer, false));
  wire(btnLeoJ2,   () => showLeo(renderer, true));
  wire(btnLaunch,  () => showLaunch(renderer));
  wire(btnHohmann, () => showHohmann(renderer));
  wire(btnLambert, () => showLambert(renderer));
  wire(btnCislunar, () => showCislunar(renderer));
  wire(btnCr3bp,    () => showCr3bp(renderer));
  wire(btnManifold, () => showLyapunovManifold(renderer));
  wire(btnWsb,      () => showWsb(renderer));

  // Maneuver-editor toggle + initial population.
  resetEditor();
  btnEditor.addEventListener("click", () => {
    editor.classList.toggle("show");
    const visible = editor.classList.contains("show");
    btnEditor.setAttribute("aria-pressed", visible ? "true" : "false");
  });
  (document.getElementById("btn-add-man") as HTMLButtonElement)
    .addEventListener("click", () => {
      const horizon = editorState.duration_s;
      const last = editorState.maneuvers[editorState.maneuvers.length - 1];
      const next_t = last ? Math.min(horizon * 0.99, last.t_offset_s + 600) : horizon * 0.25;
      editorState.maneuvers.push({
        kind: "impulsive",
        t_offset_s: next_t,
        dv_r: 0, dv_i: 50, dv_c: 0,
        duration_s: 300, thrust_n: 1000, isp_s: 300,
        dir_r: 0, dir_i: 1, dir_c: 0,
      });
      renderManeuverList();
    });
  (document.getElementById("btn-apply") as HTMLButtonElement)
    .addEventListener("click", () => {
      applyEditor(renderer).catch((e) => setStatus(`editor error: ${(e as Error).message}`));
    });
  (document.getElementById("btn-reset") as HTMLButtonElement)
    .addEventListener("click", () => resetEditor());

  const refreshVehicle = () => {
    const drag = (document.getElementById("pert-drag") as HTMLInputElement).checked;
    const srp = (document.getElementById("pert-srp") as HTMLInputElement).checked;
    (document.getElementById("pert-vehicle") as HTMLElement).style.display =
      (drag || srp) ? "block" : "none";
  };
  (document.getElementById("pert-drag") as HTMLInputElement)
    .addEventListener("change", refreshVehicle);
  (document.getElementById("pert-srp") as HTMLInputElement)
    .addEventListener("change", refreshVehicle);

  // Central body selector — repopulate the IC fields with sensible defaults
  // for the new body (low circular orbit) and refresh the perturbation panel.
  (document.getElementById("ic-body") as HTMLSelectElement)
    .addEventListener("change", (e) => {
      applyBodyDefaults((e.target as HTMLSelectElement).value as BodyName);
      refreshVehicle();
      renderSolverForm();  // refresh defaults that depend on the body
    });

  // Solver panel.
  renderSolverForm();
  (document.getElementById("solver-type") as HTMLSelectElement)
    .addEventListener("change", renderSolverForm);
  (document.getElementById("btn-solve") as HTMLButtonElement)
    .addEventListener("click", () => {
      runSolver(renderer).catch((e) => setStatus(`solver error: ${(e as Error).message}`));
    });

  await showLeo(renderer, false);

  const sock = new SimSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
    (msg) => console.debug("ws", msg),
    (s) => console.debug("ws status:", s),
  );
  sock.connect();
}

main().catch((e) => setStatus(`fatal: ${(e as Error).message}`));
