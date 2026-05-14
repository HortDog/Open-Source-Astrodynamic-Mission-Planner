import {
  cr3bpLagrange,
  cr3bpManifold,
  cr3bpPeriodicOrbit,
  cr3bpPropagate,
  cr3bpWsb,
  fetchHealth,
  launchDefaultConfig,
  optimizeHohmann,
  optimizeLambert,
  optimizeMultiBurn,
  propagate,
  propagateStream,
  runLaunch,
  runLaunchConfig,
  SimSocket,
  spiceEphemeris,
  spiceState,
  spiceStatus,
  tleByNorad,
  tleParse,
  transformStates,
  type FiniteBurnSpec,
  type LaunchConfig,
  type PropagateResponse,
  type Vec3,
} from "./api";
import {
  DEG,
  ecefToJ2000,
  elementsToState,
  j2000ToEcef,
  RAD,
  stateToElements,
  type OrbitalElements,
} from "./kepler";
import { fetchEarthCoastline } from "./render/earth-coastline";
import { gmstFromTdb } from "./render/gmst";
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
const btnCancel = document.getElementById("btn-cancel") as HTMLButtonElement;
const btnRecenter = document.getElementById("btn-recenter") as HTMLButtonElement;
const btnSaveMission = document.getElementById("btn-save-mission") as HTMLButtonElement;
const fileLoadMission = document.getElementById("file-load-mission") as HTMLInputElement;
const btnExportTraj = document.getElementById("btn-export-traj") as HTMLButtonElement;
const spiceBadge = document.getElementById("spice-badge") as HTMLSpanElement;
const chartKindSel = document.getElementById("chart-kind") as HTMLSelectElement;
const btnPlay = document.getElementById("btn-play") as HTMLButtonElement;
const playSpeedSel = document.getElementById("play-speed") as HTMLSelectElement;

const setStatus = (s: string) => { status.textContent = s; };
const showBanner = (msg: string) => { banner.textContent = msg; banner.classList.add("show"); };

// J2000 epoch in UTC milliseconds (2000-01-01T11:58:55.816Z).
const J2000_UTC_MS = Date.UTC(2000, 0, 1, 11, 58, 55, 816);
// Approximate TT − UTC offset (32.184 s + 37 s leap seconds for post-2017 dates).
// Backend SPICE module re-derives precise leap seconds; this is good to ~1 s.
const TT_UTC_OFFSET_S = 69.184;

/** Convert an ISO-ish "YYYY-MM-DDTHH:MM:SS" UTC string to TDB seconds since J2000.
 *  Returns 0 for empty input. */
function utcToTdbSeconds(utc: string): number {
  if (!utc) return 0;
  const ms = Date.parse(utc.endsWith("Z") ? utc : utc + "Z");
  if (!Number.isFinite(ms)) return 0;
  return (ms - J2000_UTC_MS) / 1000 + TT_UTC_OFFSET_S;
}

/** Inverse of utcToTdbSeconds: TDB seconds → "YYYY-MM-DDTHH:MM:SS" (UTC). */
function tdbSecondsToUtcInput(tdb: number): string {
  if (!Number.isFinite(tdb) || tdb === 0) return "";
  const utcMs = (tdb - TT_UTC_OFFSET_S) * 1000 + J2000_UTC_MS;
  const d = new Date(utcMs);
  // Keep to seconds resolution and strip the trailing "Z" / ".000" for datetime-local.
  return d.toISOString().slice(0, 19);
}

const MU_EARTH = 3.986004418e14;
const R_EARTH = 6_378_137;
const R0 = 7_000_000;
const INCLINATION_DEG = 51.6;

// Body constants table for the central-body selector. Mirrors the canonical
// values in `backend/oamp/bodies.py` — used to seed the editor IC defaults
// and to render the body as an oblate spheroid where appropriate.
type BodyName = "EARTH" | "MOON";
const BODY: Record<BodyName, {
  mu: number;
  radius: number;        // equatorial
  polar_radius: number;  // along the spin axis
  default_alt_m: number;
  label: string;
}> = {
  EARTH: {
    mu: 3.986004418e14,
    radius: 6_378_137,
    polar_radius: 6_356_752.314,  // WGS-84, 1/f = 298.257_223_563
    default_alt_m: 622_000,
    label: "Earth",
  },
  MOON: {
    mu: 4.9048695e12,
    radius: 1_737_400,
    polar_radius: 1_736_000,      // IAU 2009 — 1/f ≈ 1240.6
    default_alt_m: 100_000,
    label: "Moon",
  },
};

// Earth--Moon system constants used by the EM_SYNODIC view-frame transform.
// Mirror of backend `oamp.dynamics.cr3bp.EM_MU` / `EM_LENGTH_M` / `EM_MEAN_MOTION_RAD_S`.
const EM_MU_FRONTEND = BODY.MOON.mu / (BODY.EARTH.mu + BODY.MOON.mu);
const EM_LENGTH_M = 384_400_000;
// Non-dim CR3BP time has units of 1/n (mean motion). For the Earth--Moon
// system n ≈ 2.66e-6 rad/s, so one non-dim unit ≈ 375 700 s ≈ 4.35 days.
// Multiply non-dim t by EM_TIME_UNIT_S to get SI seconds before feeding it to
// the scrubber / playback engine.
const EM_MEAN_MOTION_RAD_S = 2.661699e-6;
const EM_TIME_UNIT_S = 1 / EM_MEAN_MOTION_RAD_S;

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

// --------------------------------------------------------------------------- //
//  Past/future trajectory split — the bit of the orbit before the current
//  scrub index renders as a solid line; everything after renders dashed.
//  Cached positions/colors are uploaded once by `paintTrajectory` and then
//  re-split on every scrub change without a re-derivation cost.
// --------------------------------------------------------------------------- //

let cachedPositions: Float32Array<ArrayBuffer> | null = null;
let cachedColors: Float32Array<ArrayBuffer> | null = null;

function drawTrajectorySplit(idx: number): void {
  if (!activeRenderer || !cachedPositions) return;
  const N = cachedPositions.length / 3;
  if (N < 2) {
    activeRenderer.drawTrajectory(cachedPositions, cachedColors ?? undefined);
    return;
  }
  // Clamp idx and define the cut. Past gets [0..idx+1), future gets [idx..N).
  // The shared vertex at `idx` glues the two halves so there is no visual gap.
  const i = Math.min(Math.max(idx, 0), N - 1);
  const past   = cachedPositions.subarray(0, (i + 1) * 3) as Float32Array<ArrayBuffer>;
  const future = cachedPositions.subarray(i * 3)         as Float32Array<ArrayBuffer>;
  const pastCols = cachedColors
    ? (cachedColors.subarray(0, (i + 1) * 3) as Float32Array<ArrayBuffer>)
    : undefined;
  const futureCols = cachedColors
    ? (cachedColors.subarray(i * 3) as Float32Array<ArrayBuffer>)
    : undefined;
  const futureArg: { positions: Float32Array<ArrayBuffer>; colors?: Float32Array<ArrayBuffer> } =
    { positions: future };
  if (futureCols) futureArg.colors = futureCols;
  activeRenderer.drawTrajectory(past, pastCols, futureArg);
}

/** Paint a full trajectory and cache the buffers so subsequent scrub /
 *  playback events can re-split without re-deriving the positions. */
function paintTrajectory(
  positions: Float32Array<ArrayBuffer>,
  colors?: Float32Array<ArrayBuffer>,
): void {
  cachedPositions = positions;
  cachedColors = colors ?? null;
  // Initial paint shows the whole orbit as solid — split kicks in on scrub /
  // play. We achieve that by drawing with idx = N − 1 (past = full, future = ∅).
  if (activeRenderer) {
    const N = positions.length / 3;
    drawTrajectorySplit(Math.max(0, N - 1));
  }
}

// --------------------------------------------------------------------------- //
//  Playback engine — animates the scrub from current position to the end of
//  the mission. `playSpeed` is sim-seconds per real-second; the binary search
//  on `activeTraj.t` lets us deal with non-uniform sample spacing (multi-arc
//  trajectories distribute samples per arc, not uniformly in time).
// --------------------------------------------------------------------------- //

let playing = false;
let playRafId: number | null = null;
let playStartReal = 0;
let playStartSimT = 0;

function setBtnPlayLabel(): void {
  btnPlay.textContent = playing ? "❚❚" : "▶";
  btnPlay.title = playing ? "Pause" : "Play mission animation";
}

function startPlayback(): void {
  if (playing || !activeTraj || activeTraj.t.length < 2) return;
  const cur = parseInt(scrub.value, 10);
  const endIdx = activeTraj.t.length - 1;
  // If parked at the end (or close to it), rewind to the start so play loops.
  const startIdx = cur >= endIdx ? 0 : cur;
  scrub.value = String(startIdx);
  updateScrubMarker(startIdx);
  playing = true;
  playStartReal = performance.now();
  playStartSimT = activeTraj.t[startIdx] ?? 0;
  setBtnPlayLabel();
  playRafId = requestAnimationFrame(tickPlayback);
}

function stopPlayback(): void {
  playing = false;
  if (playRafId !== null) cancelAnimationFrame(playRafId);
  playRafId = null;
  setBtnPlayLabel();
}

/** Finalise a demo: pick a playback speed appropriate for the mission
 *  duration, optionally open the editor sidebar, and auto-start playback so
 *  the user immediately sees an animated mission with the past-as-solid /
 *  future-as-dashed split, rotating Earth, and SPICE-driven Moon. */
function finalizeDemo(opts: {
  /** Sim seconds per real second. If omitted, picked to make the full
   *  mission play through in ~8 real seconds. */
  speed?: number;
  /** Whether to open the maneuver-editor sidebar. Default true so the user
   *  can immediately tweak the demo's IC. */
  openEditor?: boolean;
}): void {
  if (opts.openEditor !== false) {
    editor.classList.add("show");
    btnEditor.setAttribute("aria-pressed", "true");
  }
  if (activeTraj && activeTraj.t.length >= 2) {
    const t0 = activeTraj.t[0]!;
    const tN = activeTraj.t[activeTraj.t.length - 1]!;
    const duration = Math.max(tN - t0, 1);
    const speed = opts.speed ?? Math.max(1, duration / 8);  // ~8 s playback
    // Snap to the nearest preset in the dropdown so the displayed value
    // matches what's actually playing.
    const presets = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000];
    const closest = presets.reduce((a, b) =>
      Math.abs(Math.log(b) - Math.log(speed)) < Math.abs(Math.log(a) - Math.log(speed)) ? b : a);
    playSpeedSel.value = String(closest);
  }
  // Defer one frame so the renderer has uploaded the trajectory buffer.
  requestAnimationFrame(() => startPlayback());
}

function tickPlayback(now: number): void {
  if (!playing || !activeTraj) { stopPlayback(); return; }
  const speed = parseFloat(playSpeedSel.value || "1000");
  const elapsedReal = (now - playStartReal) / 1000;
  const targetSimT = playStartSimT + elapsedReal * speed;
  const tArr = activeTraj.t;
  const tEnd = tArr[tArr.length - 1]!;
  if (targetSimT >= tEnd) {
    const last = tArr.length - 1;
    scrub.value = String(last);
    updateScrubMarker(last);
    stopPlayback();
    return;
  }
  // Binary search for the largest sample index with t ≤ targetSimT.
  let lo = 0, hi = tArr.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (tArr[mid]! <= targetSimT) lo = mid; else hi = mid;
  }
  scrub.value = String(lo);
  updateScrubMarker(lo);
  playRafId = requestAnimationFrame(tickPlayback);
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
    `t=${formatSimTime(t)} · |r|=${(r / 1000).toFixed(0)} km · |v|=${(v / 1000).toFixed(2)} km/s`;
  updateAltitudeCursor(idx);
  // Repaint the trajectory with the past/future split at this scrub idx.
  if (cachedPositions) drawTrajectorySplit(idx);
  // Spin Earth's wireframe + coastline to match the current mission time.
  updateEarthRotationAt(t);
}

scrub.addEventListener("input", () => updateScrubMarker(parseInt(scrub.value, 10)));

/** Render mission time in the largest readable unit (seconds → hours → days). */
function formatSimTime(t: number): string {
  if (!Number.isFinite(t)) return "—";
  const a = Math.abs(t);
  if (a < 600)    return `${t.toFixed(0)} s`;
  if (a < 7200)   return `${(t / 60).toFixed(1)} min`;
  if (a < 172800) return `${(t / 3600).toFixed(2)} h`;
  return `${(t / 86400).toFixed(2)} d`;
}

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
  rotateEarth = true; activeEpochTdb = 0;
  if (renderer.kind !== "webgpu") return;
  renderer.setSceneScale(R0);
  renderer.setCentralBody(R_EARTH, "EARTH", BODY.EARTH.polar_radius);
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
  paintTrajectory(statesToPositions(data.states));
  setActiveTrajectory(data.t, data.states);
  setStatus(j2
    ? `LEO + J2: ${data.states.length} samples over ${(duration / 60).toFixed(0)} min — note nodal regression`
    : `LEO: ${data.states.length} samples over one period (${period.toFixed(0)} s)`);
  finalizeDemo({});
}

async function showLaunch(renderer: Renderer): Promise<void> {
  rotateEarth = true; activeEpochTdb = 0;
  if (renderer.kind !== "webgpu") return;
  renderer.setSceneScale(R_EARTH);
  renderer.setCentralBody(R_EARTH, "EARTH", BODY.EARTH.polar_radius);

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
  paintTrajectory(positions, colors);
  setActiveTrajectory(data.t, data.states);
  setStatus(
    `Launch demo: burnout T+${data.burnout_time_s.toFixed(0)}s, ` +
    `Δv_circ=${data.circularization_dv_m_s.toFixed(0)} m/s, ` +
    `final orbit ${data.final_periapsis_km.toFixed(0)} × ${data.final_apoapsis_km.toFixed(0)} km`,
  );
  finalizeDemo({});
}

/** Hohmann LEO→GEO. Three phases displayed:
 *  green (1 LEO orbit) → amber (half transfer ellipse) → cyan (1 GEO orbit). */
async function showHohmann(renderer: Renderer): Promise<void> {
  rotateEarth = true; activeEpochTdb = 0;
  if (renderer.kind !== "webgpu") return;

  const r_leo = R_EARTH + 400e3;       // 400 km LEO
  const r_geo = 42_164e3;              // GEO

  renderer.setSceneScale(r_geo);
  renderer.setCentralBody(R_EARTH, "EARTH", BODY.EARTH.polar_radius);

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
  paintTrajectory(positions, colors);
  setActiveTrajectory(data.t, data.states);

  setStatus(
    `Hohmann LEO(400 km)→GEO  ` +
    `Δv₁=${(ho.dv1_m_s / 1000).toFixed(2)} km/s · ` +
    `Δv₂=${(ho.dv2_m_s / 1000).toFixed(2)} km/s · ` +
    `Δv_total=${(ho.dv_total_m_s / 1000).toFixed(2)} km/s · ` +
    `TOF=${(ho.transfer_time_s / 3600).toFixed(2)} h`,
  );
  finalizeDemo({});
}

/** Lambert intercept: depart from a 500 km circular orbit, arrive at a target
 *  point on a 4 000 km circular orbit 120° downrange after a chosen TOF. */
async function showLambert(renderer: Renderer): Promise<void> {
  rotateEarth = true; activeEpochTdb = 0;
  if (renderer.kind !== "webgpu") return;

  const r_dep = R_EARTH + 500e3;
  const r_arr = R_EARTH + 4_000e3;
  const transfer_angle_deg = 120;
  const tof_s = 4 * 60 * 60;          // 4 h

  renderer.setSceneScale(r_arr * 1.2);
  renderer.setCentralBody(R_EARTH, "EARTH", BODY.EARTH.polar_radius);

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
  paintTrajectory(positions, colors);
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
  finalizeDemo({});
}

/** Cislunar demo: Earth-Moon Lambert transfer.
 *
 *  Loads a 200-km LEO initial state, queries the Moon's ephemeris at a fixed
 *  demo epoch via SPICE, solves Lambert for a ~3.5-day transfer to that
 *  position, applies the TLI Δv as an impulsive RIC maneuver, propagates with
 *  J2 + third-body Moon active, and renders the Moon at its inertial position.
 *  Requires SPICE kernels — bail with a hint if `/spice/state` is unavailable. */
async function showCislunar(renderer: Renderer): Promise<void> {
  rotateEarth = true; activeEpochTdb = 0;
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
  finalizeDemo({});
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
  rotateEarth = false;

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
  renderer.setCentralBody(BODY.EARTH.radius, "EARTH", BODY.EARTH.polar_radius);  // Earth at (−μ, 0)
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
  paintTrajectory(positions);

  // Build a synthetic states-with-velocity array so the scrubber can drive
  // the RIC marker and altitude chart.
  const statesSi: number[][] = traj.states.map(s => [
    s[0]! * L_M, s[1]! * L_M, s[2]! * L_M,
    s[3]! * L_M * EM_MEAN_MOTION_RAD_S,
    s[4]! * L_M * EM_MEAN_MOTION_RAD_S,
    s[5]! * L_M * EM_MEAN_MOTION_RAD_S,
  ]);
  setActiveTrajectory(traj.t.map(t => t * EM_TIME_UNIT_S), statesSi);

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
  finalizeDemo({});
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
  rotateEarth = false;

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
  renderer.setCentralBody(BODY.EARTH.radius, "EARTH", BODY.EARTH.polar_radius);

  // Earth-centric synodic view: barycentric x + μ·d_em shift.
  const xShift = EM_MU_FRONTEND * L_M;
  const orbitPositions = new Float32Array(propResp.states.length * 3);
  for (let i = 0; i < propResp.states.length; i++) {
    orbitPositions[i * 3 + 0] = propResp.states[i]![0]! * L_M + xShift;
    orbitPositions[i * 3 + 1] = propResp.states[i]![1]! * L_M;
    orbitPositions[i * 3 + 2] = propResp.states[i]![2]! * L_M;
  }
  paintTrajectory(orbitPositions);

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
  const n = EM_MEAN_MOTION_RAD_S;
  const statesSi: number[][] = propResp.states.map(s => [
    s[0]! * L_M + xShift, s[1]! * L_M, s[2]! * L_M,
    s[3]! * L_M * n, s[4]! * L_M * n, s[5]! * L_M * n,
  ]);
  setActiveTrajectory(propResp.t.map(t => t * EM_TIME_UNIT_S), statesSi);

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
  finalizeDemo({});
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
  rotateEarth = false;

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
  renderer.setCentralBody(BODY.EARTH.radius, "EARTH", BODY.EARTH.polar_radius);
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
  // WSB is a static diagnostic — no time evolution to play through, just open
  // the editor so users can see / tweak the panel inputs.
  finalizeDemo({ speed: 1 });
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
type PropMode = "J2000" | "CR3BP";
type IcForm = "cartesian" | "elements" | "ecef" | "cr3bp";
type Integrator = "dop853" | "verlet" | "yoshida4";

type EditorState = {
  body: BodyName;
  frame: ViewFrame;
  mode: PropMode;
  ic_form: IcForm;
  rx: number; ry: number; rz: number;
  vx: number; vy: number; vz: number;
  // CR3BP non-dim state (only used when mode === "CR3BP")
  cr_x: number; cr_y: number; cr_z: number;
  cr_vx: number; cr_vy: number; cr_vz: number;
  cr_tfin: number;
  cr_mu: number;
  duration_s: number;
  steps: number;
  initial_mass_kg: number;
  integrator: Integrator;
  t0_tdb: number;
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
};

const editorState: EditorState = {
  body: "EARTH",
  frame: "J2000",
  mode: "J2000",
  ic_form: "cartesian",
  rx: R0, ry: 0, rz: 0,
  vx: 0, vy: Math.sqrt(MU_EARTH / R0), vz: 0,
  cr_x: 0.836915, cr_y: 0, cr_z: 0,
  cr_vx: 0, cr_vy: 0.0, cr_vz: 0,
  cr_tfin: 6.0,
  cr_mu: 0.01215,
  duration_s: 2 * Math.PI * Math.sqrt((R0 * R0 * R0) / MU_EARTH),
  steps: 400,
  initial_mass_kg: 1000,
  integrator: "dop853",
  t0_tdb: 0,
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

// --------------------------------------------------------------------------- //
//  Preset orbit library — canonical initial conditions keyed by name.
// --------------------------------------------------------------------------- //

type PresetFn = (st: EditorState) => void;
const PRESETS: Record<string, PresetFn> = {
  iss: (s) => loadElementsPreset(s, "EARTH", { a: 6_778_137, e: 0.0003, i: 51.6, raan: 0, argp: 0, nu: 0 }),
  leo500: (s) => loadElementsPreset(s, "EARTH", { a: 6_878_137, e: 0, i: 0, raan: 0, argp: 0, nu: 0 }),
  sso600: (s) => loadElementsPreset(s, "EARTH", { a: 6_978_137, e: 0, i: 97.78, raan: 0, argp: 0, nu: 0 }),
  gto: (s) => loadElementsPreset(s, "EARTH", {
    a: (6_628_137 + 42_164_137) / 2,
    e: (42_164_137 - 6_628_137) / (42_164_137 + 6_628_137),
    i: 7, raan: 0, argp: 0, nu: 0,
  }),
  geo: (s) => loadElementsPreset(s, "EARTH", { a: 42_164_137, e: 0, i: 0, raan: 0, argp: 0, nu: 0 }),
  molniya: (s) => loadElementsPreset(s, "EARTH", {
    a: 26_600_000, e: 0.74, i: 63.4, raan: 0, argp: 270, nu: 0,
  }),
  lunar100: (s) => loadElementsPreset(s, "MOON", { a: 1_837_400, e: 0, i: 0, raan: 0, argp: 0, nu: 0 }),
  eml1_halo: (s) => {
    s.mode = "CR3BP"; s.ic_form = "cr3bp"; s.cr_mu = 0.01215;
    // Approximate L1 Lyapunov seed in CR3BP non-dim units.
    s.cr_x = 0.836915; s.cr_y = 0; s.cr_z = 0;
    s.cr_vx = 0;  s.cr_vy = 0.0; s.cr_vz = 0;
    s.cr_tfin = 6.0;
  },
  eml2_halo: (s) => {
    s.mode = "CR3BP"; s.ic_form = "cr3bp"; s.cr_mu = 0.01215;
    s.cr_x = 1.155682; s.cr_y = 0; s.cr_z = 0;
    s.cr_vx = 0; s.cr_vy = 0.0; s.cr_vz = 0;
    s.cr_tfin = 6.0;
  },
};

function loadElementsPreset(
  s: EditorState,
  body: BodyName,
  el: { a: number; e: number; i: number; raan: number; argp: number; nu: number },
): void {
  s.body = body;
  s.mode = "J2000";
  s.ic_form = "elements";
  const mu = BODY[body].mu;
  const { r, v } = elementsToState({
    a: el.a, e: el.e, i: el.i * RAD, raan: el.raan * RAD, argp: el.argp * RAD, nu: el.nu * RAD,
  }, mu);
  s.rx = r[0]; s.ry = r[1]; s.rz = r[2];
  s.vx = v[0]; s.vy = v[1]; s.vz = v[2];
  // Default duration: one orbital period (or 6 h for hyperbolae).
  if (el.a > 0 && el.e < 1) {
    s.duration_s = 2 * Math.PI * Math.sqrt((el.a * el.a * el.a) / mu);
  } else {
    s.duration_s = 6 * 3600;
  }
  s.steps = 400;
}

// --------------------------------------------------------------------------- //
//  Latest trajectory cache — used by export, recenter, and chart selector.
// --------------------------------------------------------------------------- //

let latestTrajectory: { t: number[]; states: number[][] } | null = null;
let streamAbort: AbortController | null = null;

// --------------------------------------------------------------------------- //
//  Earth-rotation state. When the rendered scene is in an inertial frame the
//  central-body wireframe should rotate at GMST; in a synodic / body-fixed
//  view the frame absorbs the spin and the wireframe stays put.
// --------------------------------------------------------------------------- //

// Mission epoch in TDB seconds since J2000 — set by applyEditor / demos when
// they kick off a propagation. The scrub-driven Earth rotation reads this plus
// the scrub-tick relative time to compute the current Greenwich angle.
let activeEpochTdb = 0;
// True when the current rendering is in an inertial frame and the body should
// visibly rotate. False for EM-synodic, CR3BP, or any rotating-frame view.
let rotateEarth = true;

function updateEarthRotationAt(t_offset_s: number): void {
  if (!activeRenderer) return;
  if (!rotateEarth) {
    activeRenderer.setCentralBodyOrientation(0);
    return;
  }
  activeRenderer.setCentralBodyOrientation(gmstFromTdb(activeEpochTdb + t_offset_s));
}

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
  (document.getElementById("ic-mode") as HTMLSelectElement).value = editorState.mode;
  (document.getElementById("ic-form") as HTMLSelectElement).value = editorState.ic_form;
  (document.getElementById("ic-rx") as HTMLInputElement).value = String(editorState.rx);
  (document.getElementById("ic-ry") as HTMLInputElement).value = String(editorState.ry);
  (document.getElementById("ic-rz") as HTMLInputElement).value = String(editorState.rz);
  (document.getElementById("ic-vx") as HTMLInputElement).value = String(editorState.vx);
  (document.getElementById("ic-vy") as HTMLInputElement).value = String(editorState.vy);
  (document.getElementById("ic-vz") as HTMLInputElement).value = String(editorState.vz);
  (document.getElementById("ic-duration") as HTMLInputElement).value = editorState.duration_s.toFixed(0);
  (document.getElementById("ic-steps") as HTMLInputElement).value = String(editorState.steps);
  (document.getElementById("ic-mass") as HTMLInputElement).value = String(editorState.initial_mass_kg);
  (document.getElementById("ic-integrator") as HTMLSelectElement).value = editorState.integrator;
  (document.getElementById("ic-epoch") as HTMLInputElement).value = tdbSecondsToUtcInput(editorState.t0_tdb);
  (document.getElementById("cr-x") as HTMLInputElement).value  = String(editorState.cr_x);
  (document.getElementById("cr-y") as HTMLInputElement).value  = String(editorState.cr_y);
  (document.getElementById("cr-z") as HTMLInputElement).value  = String(editorState.cr_z);
  (document.getElementById("cr-vx") as HTMLInputElement).value = String(editorState.cr_vx);
  (document.getElementById("cr-vy") as HTMLInputElement).value = String(editorState.cr_vy);
  (document.getElementById("cr-vz") as HTMLInputElement).value = String(editorState.cr_vz);
  (document.getElementById("cr-tfin") as HTMLInputElement).value = String(editorState.cr_tfin);
  (document.getElementById("cr-mu") as HTMLInputElement).value = String(editorState.cr_mu);
  // Populate the elements + ECEF panels from the current Cartesian state so
  // every lens is in sync at all times.
  fillElementsFromState();
  fillEcefFromState();
  updateIcFormVisibility();
}

function fillElementsFromState(): void {
  try {
    const mu = BODY[editorState.body].mu;
    const el = stateToElements(
      [editorState.rx, editorState.ry, editorState.rz],
      [editorState.vx, editorState.vy, editorState.vz],
      mu,
    );
    (document.getElementById("ic-sma") as HTMLInputElement).value = (el.a / 1000).toFixed(3);
    (document.getElementById("ic-ecc") as HTMLInputElement).value = el.e.toFixed(6);
    (document.getElementById("ic-inc") as HTMLInputElement).value = (el.i * DEG).toFixed(3);
    (document.getElementById("ic-raan") as HTMLInputElement).value = (el.raan * DEG).toFixed(3);
    (document.getElementById("ic-argp") as HTMLInputElement).value = (el.argp * DEG).toFixed(3);
    (document.getElementById("ic-nu") as HTMLInputElement).value = (el.nu * DEG).toFixed(3);
  } catch {
    /* singular state — leave elements panel as-is */
  }
}

function readElementsToState(): void {
  const n = (id: string) => parseFloat((document.getElementById(id) as HTMLInputElement).value);
  const el: OrbitalElements = {
    a:    n("ic-sma") * 1000,
    e:    n("ic-ecc"),
    i:    n("ic-inc") * RAD,
    raan: n("ic-raan") * RAD,
    argp: n("ic-argp") * RAD,
    nu:   n("ic-nu") * RAD,
  };
  if (!Number.isFinite(el.a) || !Number.isFinite(el.e)) return;
  const mu = BODY[editorState.body].mu;
  const { r, v } = elementsToState(el, mu);
  editorState.rx = r[0]; editorState.ry = r[1]; editorState.rz = r[2];
  editorState.vx = v[0]; editorState.vy = v[1]; editorState.vz = v[2];
  // Sync the Cartesian inputs so the user sees the result.
  (document.getElementById("ic-rx") as HTMLInputElement).value = String(r[0]);
  (document.getElementById("ic-ry") as HTMLInputElement).value = String(r[1]);
  (document.getElementById("ic-rz") as HTMLInputElement).value = String(r[2]);
  (document.getElementById("ic-vx") as HTMLInputElement).value = String(v[0]);
  (document.getElementById("ic-vy") as HTMLInputElement).value = String(v[1]);
  (document.getElementById("ic-vz") as HTMLInputElement).value = String(v[2]);
}

function fillEcefFromState(): void {
  const gmst = gmstFromTdb(editorState.t0_tdb);
  const { r, v } = j2000ToEcef(
    [editorState.rx, editorState.ry, editorState.rz],
    [editorState.vx, editorState.vy, editorState.vz],
    gmst,
  );
  (document.getElementById("ic-erx") as HTMLInputElement).value = r[0].toFixed(3);
  (document.getElementById("ic-ery") as HTMLInputElement).value = r[1].toFixed(3);
  (document.getElementById("ic-erz") as HTMLInputElement).value = r[2].toFixed(3);
  (document.getElementById("ic-evx") as HTMLInputElement).value = v[0].toFixed(6);
  (document.getElementById("ic-evy") as HTMLInputElement).value = v[1].toFixed(6);
  (document.getElementById("ic-evz") as HTMLInputElement).value = v[2].toFixed(6);
}

function readEcefToState(): void {
  const n = (id: string) => parseFloat((document.getElementById(id) as HTMLInputElement).value);
  const gmst = gmstFromTdb(editorState.t0_tdb);
  const { r, v } = ecefToJ2000(
    [n("ic-erx"), n("ic-ery"), n("ic-erz")],
    [n("ic-evx"), n("ic-evy"), n("ic-evz")],
    gmst,
  );
  if (![...r, ...v].every(Number.isFinite)) return;
  editorState.rx = r[0]; editorState.ry = r[1]; editorState.rz = r[2];
  editorState.vx = v[0]; editorState.vy = v[1]; editorState.vz = v[2];
  // Mirror into the Cartesian inputs so the user can flip to that view.
  (document.getElementById("ic-rx") as HTMLInputElement).value = String(r[0]);
  (document.getElementById("ic-ry") as HTMLInputElement).value = String(r[1]);
  (document.getElementById("ic-rz") as HTMLInputElement).value = String(r[2]);
  (document.getElementById("ic-vx") as HTMLInputElement).value = String(v[0]);
  (document.getElementById("ic-vy") as HTMLInputElement).value = String(v[1]);
  (document.getElementById("ic-vz") as HTMLInputElement).value = String(v[2]);
}

function updateIcFormVisibility(): void {
  const form = editorState.ic_form;
  (document.getElementById("ic-cartesian") as HTMLElement).style.display = form === "cartesian" ? "block" : "none";
  (document.getElementById("ic-elements")  as HTMLElement).style.display = form === "elements"  ? "block" : "none";
  (document.getElementById("ic-ecef")      as HTMLElement).style.display = form === "ecef"      ? "block" : "none";
  (document.getElementById("ic-cr3bp")     as HTMLElement).style.display = form === "cr3bp"     ? "block" : "none";
}

function readIc(): void {
  const n = (id: string) => parseFloat((document.getElementById(id) as HTMLInputElement).value);
  editorState.body = (document.getElementById("ic-body") as HTMLSelectElement).value as BodyName;
  editorState.frame = (document.getElementById("ic-frame") as HTMLSelectElement).value as ViewFrame;
  editorState.mode = (document.getElementById("ic-mode") as HTMLSelectElement).value as PropMode;
  editorState.ic_form = (document.getElementById("ic-form") as HTMLSelectElement).value as IcForm;

  // Read the epoch first — ECEF conversion depends on it.
  editorState.t0_tdb = utcToTdbSeconds((document.getElementById("ic-epoch") as HTMLInputElement).value);

  if (editorState.ic_form === "elements") {
    readElementsToState();
  } else if (editorState.ic_form === "ecef") {
    readEcefToState();
  } else if (editorState.ic_form === "cr3bp") {
    editorState.cr_x = n("cr-x"); editorState.cr_y = n("cr-y"); editorState.cr_z = n("cr-z");
    editorState.cr_vx = n("cr-vx"); editorState.cr_vy = n("cr-vy"); editorState.cr_vz = n("cr-vz");
    editorState.cr_tfin = n("cr-tfin");
    editorState.cr_mu = n("cr-mu");
  } else {
    editorState.rx = n("ic-rx"); editorState.ry = n("ic-ry"); editorState.rz = n("ic-rz");
    editorState.vx = n("ic-vx"); editorState.vy = n("ic-vy"); editorState.vz = n("ic-vz");
  }
  editorState.duration_s = n("ic-duration");
  editorState.steps = parseInt((document.getElementById("ic-steps") as HTMLInputElement).value, 10);
  editorState.initial_mass_kg = n("ic-mass");
  editorState.integrator = (document.getElementById("ic-integrator") as HTMLSelectElement).value as Integrator;
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
      updateDvBudget();
      renderEventTimeline();
    });
  });

  list.querySelectorAll("button[data-rm]").forEach((el) => {
    el.addEventListener("click", () => {
      const i = parseInt(el.getAttribute("data-rm")!, 10);
      editorState.maneuvers.splice(i, 1);
      renderManeuverList();
    });
  });

  updateDvBudget();
  renderEventTimeline();
}

// --------------------------------------------------------------------------- //
//  Δv budget pill + event timeline.
// --------------------------------------------------------------------------- //

function computeDvBudget(): { impulse: number; finite: number; total: number } {
  let imp = 0;
  let fin = 0;
  for (const m of editorState.maneuvers) {
    if (m.kind === "impulsive") {
      imp += Math.hypot(m.dv_r, m.dv_i, m.dv_c);
    } else {
      // Approximate finite-burn Δv via the rocket equation:
      //   Δv = Isp · g0 · ln(m0 / (m0 − ṁ·τ))
      const g0 = 9.80665;
      const mdot = m.thrust_n / (g0 * m.isp_s);
      const m0 = Math.max(editorState.initial_mass_kg, 1);
      const mf = Math.max(m0 - mdot * m.duration_s, 1);
      fin += m.isp_s * g0 * Math.log(m0 / mf);
    }
  }
  return { impulse: imp, finite: fin, total: imp + fin };
}

function updateDvBudget(): void {
  const pill = document.getElementById("dv-budget");
  if (!pill) return;
  const { total, impulse, finite } = computeDvBudget();
  const parts: string[] = [];
  if (impulse > 0) parts.push(`imp ${(impulse / 1000).toFixed(3)}`);
  if (finite > 0)  parts.push(`fin ${(finite / 1000).toFixed(3)}`);
  pill.textContent = parts.length
    ? `Σ|Δv| = ${(total / 1000).toFixed(3)} km/s (${parts.join(" · ")})`
    : "Σ|Δv| = 0 m/s";
}

function renderEventTimeline(): void {
  const svg = document.getElementById("event-tl") as unknown as SVGSVGElement | null;
  if (!svg) return;
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const W = svg.clientWidth || 320;
  const H = 22;
  const duration = Math.max(editorState.duration_s, 1);
  const tToX = (t: number): number => Math.min(W - 2, Math.max(2, (t / duration) * W));

  // Background axis.
  const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
  axis.setAttribute("x1", "0"); axis.setAttribute("x2", String(W));
  axis.setAttribute("y1", String(H / 2)); axis.setAttribute("y2", String(H / 2));
  axis.setAttribute("stroke", "#1a3a1a"); axis.setAttribute("stroke-width", "1");
  svg.appendChild(axis);

  for (const m of editorState.maneuvers) {
    if (m.kind === "finite") {
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", String(tToX(m.t_offset_s)));
      rect.setAttribute("y", "2");
      rect.setAttribute("width", String(Math.max(2, tToX(m.t_offset_s + m.duration_s) - tToX(m.t_offset_s))));
      rect.setAttribute("height", String(H - 4));
      rect.setAttribute("fill", "rgba(255,40,20,0.55)");
      rect.setAttribute("stroke", "#ff5030");
      svg.appendChild(rect);
    } else {
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("cx", String(tToX(m.t_offset_s)));
      c.setAttribute("cy", String(H / 2));
      c.setAttribute("r", "4");
      c.setAttribute("fill", "#80ff60");
      c.setAttribute("stroke", "#020a02");
      svg.appendChild(c);
    }
  }
}

// --------------------------------------------------------------------------- //
//  Inline validation (periapsis below surface, maneuver past horizon, etc.).
// --------------------------------------------------------------------------- //

function validateIc(): void {
  const warnEl = document.getElementById("ic-warnings");
  if (!warnEl) return;
  const warnings: string[] = [];
  const b = BODY[editorState.body];
  const rmag = Math.hypot(editorState.rx, editorState.ry, editorState.rz);
  if (editorState.mode === "J2000" && rmag < b.radius) {
    warnings.push(`|r| = ${(rmag / 1000).toFixed(0)} km is below ${b.label} radius (${(b.radius / 1000).toFixed(0)} km)`);
  }
  // Periapsis check via specific energy.
  if (editorState.mode === "J2000" && rmag > 0) {
    try {
      const el = stateToElements(
        [editorState.rx, editorState.ry, editorState.rz],
        [editorState.vx, editorState.vy, editorState.vz],
        b.mu,
      );
      if (el.e < 1) {
        const rp = el.a * (1 - el.e);
        if (rp < b.radius) {
          warnings.push(`periapsis ${(rp / 1000).toFixed(0)} km is below body surface`);
        }
      }
    } catch { /* singular — skip */ }
  }
  for (const m of editorState.maneuvers) {
    if (m.t_offset_s > editorState.duration_s) {
      warnings.push(`maneuver at t=${m.t_offset_s.toFixed(0)} s occurs after duration (${editorState.duration_s.toFixed(0)} s)`);
    }
    if (m.kind === "finite" && m.t_offset_s + m.duration_s > editorState.duration_s) {
      warnings.push(`finite burn at t=${m.t_offset_s.toFixed(0)} s extends past duration`);
    }
  }
  warnEl.textContent = warnings.length ? "⚠ " + warnings.join(" · ") : "";
}

// --------------------------------------------------------------------------- //
//  Mission persistence — save/load JSON, autosave to localStorage.
// --------------------------------------------------------------------------- //

const LS_KEY = "oamp.mission.v1";

function snapshotMission(): EditorState {
  return JSON.parse(JSON.stringify(editorState));
}

function applyMission(m: Partial<EditorState>): void {
  // Defensive copy + merge — accept any subset of fields.
  Object.assign(editorState, m);
  fillIc();
  fillPerturbations();
  renderManeuverList();
  validateIc();
}

function saveMissionFile(): void {
  const json = JSON.stringify(snapshotMission(), null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `oamp-mission-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

async function loadMissionFile(file: File): Promise<void> {
  const text = await file.text();
  const parsed = JSON.parse(text) as Partial<EditorState>;
  applyMission(parsed);
}

function autosaveMission(): void {
  try { localStorage.setItem(LS_KEY, JSON.stringify(snapshotMission())); }
  catch { /* quota / SSR / private-mode — ignore */ }
}

function restoreAutosave(): boolean {
  try {
    const txt = localStorage.getItem(LS_KEY);
    if (!txt) return false;
    const parsed = JSON.parse(txt) as Partial<EditorState>;
    Object.assign(editorState, parsed);
    return true;
  } catch { return false; }
}

// --------------------------------------------------------------------------- //
//  Trajectory export (CSV).
// --------------------------------------------------------------------------- //

function exportTrajectoryCsv(): void {
  if (!latestTrajectory || latestTrajectory.t.length === 0) {
    setStatus("nothing to export — run a propagation first");
    return;
  }
  const { t, states } = latestTrajectory;
  const ncol = (states[0]?.length ?? 6);
  const header = ["t_s", "rx_m", "ry_m", "rz_m", "vx_m_s", "vy_m_s", "vz_m_s"].slice(0, 1 + ncol);
  const lines: string[] = [header.join(",")];
  for (let i = 0; i < t.length; i++) {
    const row = [t[i]!.toString(), ...(states[i] ?? []).map((x) => x.toString())];
    lines.push(row.join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `oamp-traj-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// --------------------------------------------------------------------------- //
//  Footer chart selector — switches between |r|, |v|, specific energy.
// --------------------------------------------------------------------------- //

function redrawFooterChart(): void {
  if (!latestTrajectory || latestTrajectory.t.length < 2) return;
  const kind = chartKindSel?.value || "alt";
  const { t, states } = latestTrajectory;
  const mu = BODY[editorState.body].mu;
  let y: number[];
  let label: string;
  if (kind === "speed") {
    y = states.map((s) => Math.hypot(s[3] ?? 0, s[4] ?? 0, s[5] ?? 0));
    label = "|v|";
  } else if (kind === "energy") {
    y = states.map((s) => {
      const r = Math.hypot(s[0]!, s[1]!, s[2]!);
      const v2 = (s[3] ?? 0) ** 2 + (s[4] ?? 0) ** 2 + (s[5] ?? 0) ** 2;
      return v2 / 2 - mu / r;
    });
    label = "ε";
  } else {
    y = states.map((s) => Math.hypot(s[0]!, s[1]!, s[2]!));
    label = "|r|";
  }
  drawSeries(t, y, label);
}

function drawSeries(times: number[], values: number[], label: string): void {
  const line = document.getElementById("alt-line") as unknown as SVGPolylineElement | null;
  const lbl = document.getElementById("alt-label");
  if (!line || values.length < 2) return;
  altMinR = Math.min(...values);
  altMaxR = Math.max(...values);
  const pad = Math.max((altMaxR - altMinR) * 0.05, Math.abs(altMaxR) * 1e-6, 1);
  const yLo = altMinR - pad, yHi = altMaxR + pad;
  const t0 = times[0]!, tN = times[times.length - 1]!;
  const tSpan = Math.max(tN - t0, 1);
  const pts: string[] = new Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const x = ((times[i]! - t0) / tSpan) * ALT_W;
    const yy = ALT_H - ((values[i]! - yLo) / (yHi - yLo)) * ALT_H;
    pts[i] = `${x.toFixed(1)},${yy.toFixed(1)}`;
  }
  line.setAttribute("points", pts.join(" "));
  if (lbl) {
    const fmt = (v: number) => Math.abs(v) > 1e6 ? (v / 1e6).toFixed(2) + "M" : (v / 1000).toFixed(0) + "k";
    lbl.textContent = `${label}: ${fmt(altMinR)}…${fmt(altMaxR)}`;
  }
}

async function applyEditor(renderer: Renderer): Promise<void> {
  readIc();
  readPerturbations();
  validateIc();
  const b = BODY[editorState.body];

  // Drive the Earth rotation: synodic / CR3BP views freeze it (the frame
  // absorbs the spin), inertial views track GMST. Record the mission epoch
  // so the scrub handler can compute θ_G at any future tick.
  activeEpochTdb = editorState.t0_tdb;
  rotateEarth = editorState.mode === "J2000" && editorState.frame === "J2000";

  // ---- CR3BP path: non-dimensional propagation through /cr3bp/propagate. ----
  if (editorState.mode === "CR3BP") {
    setStatus("propagating CR3BP…");
    const t_span: [number, number] = [0, editorState.cr_tfin];
    const res = await cr3bpPropagate({
      state: [editorState.cr_x, editorState.cr_y, editorState.cr_z,
              editorState.cr_vx, editorState.cr_vy, editorState.cr_vz],
      t_span,
      mu: editorState.cr_mu,
      steps: Math.max(2, Math.min(20000, editorState.steps)),
    });
    // Render in non-dim units scaled to the Earth–Moon distance so the scene
    // matches the J2000 frame visually. Velocities are also scaled (the scrub
    // marker uses them for the RIC frame, and the playback engine needs t in
    // SI seconds — convert non-dim t by EM_TIME_UNIT_S).
    const L = EM_LENGTH_M;
    const n = EM_MEAN_MOTION_RAD_S;
    const states = res.states.map((s) => [
      s[0]! * L, s[1]! * L, s[2]! * L,
      s[3]! * L * n, s[4]! * L * n, s[5]! * L * n,
    ]);
    const t_si = res.t.map((t) => t * EM_TIME_UNIT_S);
    latestTrajectory = { t: t_si, states };
    renderer.setSceneScale(L * 1.5);
    renderer.setCentralBody(BODY.EARTH.radius, "EARTH", BODY.EARTH.polar_radius);
    renderer.setSecondaryBody([L, 0, 0], BODY.MOON.radius);
    moonTrackT = null; moonTrackR = null; latestMoonR = [L, 0, 0];
    paintTrajectory(statesToPositions(states));
    setActiveTrajectory(t_si, states);
    refreshCameraTarget();
    const cMin = Math.min(...res.jacobi);
    const cMax = Math.max(...res.jacobi);
    setStatus(`CR3BP: ${states.length} samples · Jacobi ${cMin.toFixed(4)}…${cMax.toFixed(4)} (Δ ${(cMax-cMin).toExponential(2)})`);
    redrawFooterChart();
    autosaveMission();
    return;
  }

  const propReq = {
    state: {
      r: [editorState.rx, editorState.ry, editorState.rz] as [number, number, number],
      v: [editorState.vx, editorState.vy, editorState.vz] as [number, number, number],
    },
    duration_s: editorState.duration_s,
    steps: editorState.steps,
    body_name: editorState.body,
    mu: b.mu,
    body_radius: b.radius,
    integrator: editorState.integrator,
    ...(editorState.t0_tdb !== 0 ? { t0_tdb: editorState.t0_tdb } : {}),
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
  };

  const useStream = (document.getElementById("ic-stream") as HTMLInputElement).checked
    || editorState.steps > 2000;

  let data: PropagateResponse;

  if (useStream) {
    setStatus("streaming propagation…");
    // Forget the cached past/future split — the intermediate stream renders go
    // straight to the renderer, and a scrub during streaming should not pull
    // from the previous trajectory.
    cachedPositions = null;
    cachedColors = null;
    streamAbort = new AbortController();
    btnCancel.classList.add("show");
    const allT: number[] = [];
    const allStates: number[][] = [];
    let perturbations: string[] = [];
    try {
      for await (const chunk of propagateStream(propReq, streamAbort.signal)) {
        if ("error" in chunk) throw new Error(chunk.error);
        if (chunk.done) { perturbations = chunk.perturbations; break; }
        allT.push(...chunk.t);
        allStates.push(...chunk.states);
        renderer.drawTrajectory(statesToPositions(allStates));
        setStatus(`streaming… ${chunk.received_steps} / ${chunk.total_steps} steps`);
        await new Promise<void>((res) => requestAnimationFrame(() => res()));
      }
    } catch (e) {
      if ((e as DOMException).name === "AbortError") {
        setStatus(`cancelled at ${allT.length} steps — partial trajectory retained`);
      } else {
        throw e;
      }
    } finally {
      btnCancel.classList.remove("show");
      streamAbort = null;
    }
    data = { t: allT, states: allStates, perturbations };
  } else {
    setStatus("propagating editor scenario…");
    data = await propagate(propReq);
  }

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
  renderer.setCentralBody(b.radius, editorState.body, b.polar_radius);

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
  paintTrajectory(statesToPositions(renderedStates), colors);
  setActiveTrajectory(data.t, renderedStates);
  latestTrajectory = { t: data.t, states: renderedStates };
  redrawFooterChart();
  autosaveMission();

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
  editorState.mode = "J2000";
  editorState.ic_form = "cartesian";
  editorState.steps = 400;
  editorState.maneuvers = [];
  editorState.initial_mass_kg = 1000;
  editorState.integrator = "dop853";
  editorState.t0_tdb = 0;
  editorState.j2 = false; editorState.jn_max = 2;
  editorState.drag = false; editorState.drag_model = "exponential";
  editorState.drag_mass_kg = 500; editorState.drag_area_m2 = 4.0; editorState.drag_cd = 2.2;
  editorState.srp = false; editorState.srp_area_m2 = 4.0; editorState.srp_cr = 1.5;
  editorState.third_body_moon = false; editorState.third_body_sun = false;
  applyBodyDefaults("EARTH");
  renderManeuverList();
  validateIc();
}

// --------------------------------------------------------------------------- //
//  Lyapunov + manifold panel (in-editor).
// --------------------------------------------------------------------------- //

async function runLyapunovPanel(renderer: Renderer): Promise<void> {
  const n = (id: string) => parseFloat((document.getElementById(id) as HTMLInputElement).value);
  const s = (id: string) => (document.getElementById(id) as HTMLSelectElement).value;
  const Lpt = parseInt(s("lyap-L"), 10) as 1 | 2;
  const Ax = n("lyap-Ax");
  const dir = s("lyap-dir") as "stable" | "unstable";
  const branch = s("lyap-branch") as "+" | "-";
  const duration = n("lyap-dur");
  const n_samples = parseInt((document.getElementById("lyap-n") as HTMLInputElement).value, 10);
  const info = document.getElementById("lyap-info")!;

  info.textContent = "computing Lyapunov orbit…";
  const orb = await cr3bpPeriodicOrbit({ family: "lyapunov", L_point: Lpt, Ax, mu: editorState.cr_mu });
  info.textContent = `orbit period ${orb.period.toFixed(3)} (DC res ${orb.dc_residual.toExponential(1)}) · computing manifold…`;
  const man = await cr3bpManifold({
    orbit_state: orb.state0,
    period: orb.period,
    mu: editorState.cr_mu,
    direction: dir,
    branch,
    n_samples,
    duration,
  });

  // Render orbit + manifold tubes scaled to the Earth–Moon distance.
  const L = EM_LENGTH_M;
  const orbitProp = await cr3bpPropagate({
    state: orb.state0, t_span: [0, orb.period], mu: editorState.cr_mu, steps: 400,
  });
  const nMM = EM_MEAN_MOTION_RAD_S;
  const orbitStates = orbitProp.states.map((p) => [
    p[0]! * L, p[1]! * L, p[2]! * L,
    p[3]! * L * nMM, p[4]! * L * nMM, p[5]! * L * nMM,
  ]);
  const tubes = man.trajectories.map((tube) => {
    const out = new Float32Array(tube.length * 3);
    for (let i = 0; i < tube.length; i++) {
      const p = tube[i]!;
      out[i * 3 + 0] = p[0]! * L;
      out[i * 3 + 1] = p[1]! * L;
      out[i * 3 + 2] = p[2]! * L;
    }
    return out;
  });
  const orbitT_si = orbitProp.t.map((t) => t * EM_TIME_UNIT_S);
  renderer.setSceneScale(L * 1.5);
  renderer.setCentralBody(BODY.EARTH.radius, "EARTH", BODY.EARTH.polar_radius);
  renderer.setSecondaryBody([L, 0, 0], BODY.MOON.radius);
  paintTrajectory(statesToPositions(orbitStates));
  renderer.setManifoldTubes(tubes);
  setActiveTrajectory(orbitT_si, orbitStates);
  latestTrajectory = { t: orbitT_si, states: orbitStates };
  info.textContent =
    `L${Lpt} Lyapunov: T=${orb.period.toFixed(3)}, J=${orb.jacobi.toFixed(4)} · ${tubes.length} manifold tubes (${dir} ${branch})`;
}

// --------------------------------------------------------------------------- //
//  WSB diagnostic panel (in-editor).
// --------------------------------------------------------------------------- //

async function runWsbPanel(renderer: Renderer): Promise<void> {
  const n = (id: string) => parseFloat((document.getElementById(id) as HTMLInputElement).value);
  const ni = (id: string) => parseInt((document.getElementById(id) as HTMLInputElement).value, 10);
  const info = document.getElementById("wsb-info")!;
  const altMin = n("wsb-alt-min") * 1000;
  const altMax = n("wsb-alt-max") * 1000;
  const nAlt = ni("wsb-alt-n");
  const nAng = ni("wsb-ang-n");
  const dur = n("wsb-dur");
  const esc = n("wsb-esc");

  const altitudes_m: number[] = [];
  for (let i = 0; i < nAlt; i++) altitudes_m.push(altMin + (altMax - altMin) * (i / Math.max(nAlt - 1, 1)));
  const angles_rad: number[] = [];
  for (let i = 0; i < nAng; i++) angles_rad.push((2 * Math.PI) * (i / nAng));

  info.textContent = `computing ${nAlt}×${nAng}=${nAlt*nAng}-cell WSB grid…`;
  const grid = await cr3bpWsb({
    altitudes_m, angles_rad, mu: editorState.cr_mu, duration: dur, escape_radius: esc,
  });

  // Build a scatter trajectory (one point per captured cell, second cloud for escaped).
  const L = EM_LENGTH_M;
  const moonX = (1 - editorState.cr_mu) * L;
  const moonR_phys = 1737_400;  // m
  const moonR_nondim = moonR_phys / L;
  const captured: number[] = [];
  const escaped: number[] = [];
  for (let i = 0; i < altitudes_m.length; i++) {
    const rAlt_m = moonR_phys + altitudes_m[i]!;
    const r_nondim = (rAlt_m / L) + moonR_nondim;
    for (let j = 0; j < angles_rad.length; j++) {
      const ang = angles_rad[j]!;
      const x = moonX + r_nondim * Math.cos(ang) * L;
      const y = r_nondim * Math.sin(ang) * L;
      const val = grid.grid[i]?.[j] ?? 0;
      if (val === 1) captured.push(x, y, 0);
      else if (val === -1) escaped.push(x, y, 0);
    }
  }
  renderer.setSceneScale(L * 1.5);
  renderer.setCentralBody(BODY.EARTH.radius, "EARTH", BODY.EARTH.polar_radius);
  renderer.setSecondaryBody([L, 0, 0], BODY.MOON.radius);
  if (captured.length > 0) {
    renderer.drawTrajectory(new Float32Array(captured));
  }
  info.textContent =
    `captured ${captured.length / 3} · escaped ${escaped.length / 3} · ` +
    `total ${altitudes_m.length * angles_rad.length}`;
}

// --------------------------------------------------------------------------- //
//  Launch config panel (in-editor).
// --------------------------------------------------------------------------- //

async function fillLaunchDefaults(): Promise<void> {
  try {
    const cfg = await launchDefaultConfig();
    (document.getElementById("lc-dry") as HTMLInputElement).value = String(cfg.vehicle.dry_mass_kg);
    (document.getElementById("lc-prop") as HTMLInputElement).value = String(cfg.vehicle.prop_mass_kg);
    (document.getElementById("lc-thrust") as HTMLInputElement).value = String(cfg.vehicle.thrust_n);
    (document.getElementById("lc-isp") as HTMLInputElement).value = String(cfg.vehicle.isp_s);
    (document.getElementById("lc-darea") as HTMLInputElement).value = String(cfg.vehicle.drag_area_m2 ?? 10);
    (document.getElementById("lc-cd") as HTMLInputElement).value = String(cfg.vehicle.drag_cd ?? 0.3);
    (document.getElementById("lc-pstart") as HTMLInputElement).value = String(cfg.pitch_start_alt_m ?? 1500);
    (document.getElementById("lc-ptarget") as HTMLInputElement).value = String(cfg.pitch_target_alt_m ?? 100_000);
    (document.getElementById("lc-pdeg") as HTMLInputElement).value = String(cfg.pitch_target_deg ?? 88);
    (document.getElementById("lc-coast") as HTMLInputElement).value = String(cfg.coast_after_burnout_s ?? 200);
  } catch (e) {
    (document.getElementById("lc-info") as HTMLElement).textContent =
      `defaults unavailable: ${(e as Error).message}`;
  }
}

async function runLaunchPanel(renderer: Renderer): Promise<void> {
  const n = (id: string) => parseFloat((document.getElementById(id) as HTMLInputElement).value);
  const info = document.getElementById("lc-info")!;
  info.textContent = "running launch sim…";
  const cfg: LaunchConfig = {
    vehicle: {
      dry_mass_kg: n("lc-dry"),
      prop_mass_kg: n("lc-prop"),
      thrust_n: n("lc-thrust"),
      isp_s: n("lc-isp"),
      drag_area_m2: n("lc-darea"),
      drag_cd: n("lc-cd"),
    },
    pitch_start_alt_m: n("lc-pstart"),
    pitch_target_alt_m: n("lc-ptarget"),
    pitch_target_deg: n("lc-pdeg"),
    coast_after_burnout_s: n("lc-coast"),
  };
  const res = await runLaunchConfig(cfg);
  const positions = statesToPositions(res.states);
  renderer.setSceneScale(Math.max(BODY.EARTH.radius * 1.2, 1e7));
  renderer.setCentralBody(BODY.EARTH.radius, "EARTH", BODY.EARTH.polar_radius);
  renderer.setSecondaryBody(null, 0);
  paintTrajectory(positions);
  setActiveTrajectory(res.t, res.states);
  latestTrajectory = { t: res.t, states: res.states };
  redrawFooterChart();
  info.textContent =
    `burnout @ ${res.burnout_time_s.toFixed(0)}s · circ Δv ${res.circularization_dv_m_s.toFixed(1)} m/s · ` +
    `${res.final_periapsis_km.toFixed(0)}×${res.final_apoapsis_km.toFixed(0)} km`;
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
      // NLP returns inertial Δvs at each epoch. Convert each to the RIC frame
      // by propagating the *no-burn* initial state to that epoch and rotating
      // through the local R̂/Î/Ĉ basis. This lets the editor speak its native
      // RIC dialect and the maneuver list re-edits correctly.
      editorState.maneuvers = [];
      const noBurn = await propagate({
        state: {
          r: [editorState.rx, editorState.ry, editorState.rz],
          v: [editorState.vx, editorState.vy, editorState.vz],
        },
        duration_s: Math.max(...epochs, tfin),
        steps: Math.max(200, epochs.length * 50),
        body_name: editorState.body,
        mu: b.mu,
        body_radius: b.radius,
      });
      for (let k = 0; k < epochs.length; k++) {
        const t = epochs[k]!;
        // Find nearest sample to epoch t.
        let lo = 0, hi = noBurn.t.length - 1;
        while (lo + 1 < hi) {
          const mid = (lo + hi) >> 1;
          if (noBurn.t[mid]! <= t) lo = mid; else hi = mid;
        }
        const s = noBurn.states[lo]!;
        const rx = s[0]!, ry = s[1]!, rz = s[2]!;
        const vx = s[3]!, vy = s[4]!, vz = s[5]!;
        const rmag = Math.hypot(rx, ry, rz);
        const Rhat: Vec3 = [rx / rmag, ry / rmag, rz / rmag];
        // C = r × v / |r×v|
        const cx = ry * vz - rz * vy;
        const cy = rz * vx - rx * vz;
        const cz = rx * vy - ry * vx;
        const cm = Math.hypot(cx, cy, cz);
        const Chat: Vec3 = [cx / cm, cy / cm, cz / cm];
        // I = C × R
        const Ihat: Vec3 = [
          Chat[1] * Rhat[2] - Chat[2] * Rhat[1],
          Chat[2] * Rhat[0] - Chat[0] * Rhat[2],
          Chat[0] * Rhat[1] - Chat[1] * Rhat[0],
        ];
        const dv = res.dv_inertial_m_s[k]!;
        const dvR = dv[0]! * Rhat[0] + dv[1]! * Rhat[1] + dv[2]! * Rhat[2];
        const dvI = dv[0]! * Ihat[0] + dv[1]! * Ihat[1] + dv[2]! * Ihat[2];
        const dvC = dv[0]! * Chat[0] + dv[1]! * Chat[1] + dv[2]! * Chat[2];
        editorState.maneuvers.push(makeImpulse(t, [dvR, dvI, dvC]));
      }
      info.innerHTML =
        `${res.converged ? "✓ converged" : "✗ diverged"} (${res.iterations} iter)<br>` +
        `Σ |Δv| = ${fmt(res.total_dv_m_s)} km/s · ${epochs.length} burns auto-loaded as RIC`;
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

  // Natural Earth coastline — fetched once at boot, non-blocking. The
  // graticule already renders synchronously inside the renderer, so demos
  // come up immediately; the coastline pops in when this resolves.
  fetchEarthCoastline()
    .then((polys) => renderer.setEarthCoastline(polys))
    .catch((e) => console.warn("coastline fetch failed:", (e as Error).message));

  // SPICE kernel status badge — polled once at boot.
  try {
    const st = await spiceStatus();
    if (st.error) {
      spiceBadge.classList.remove("pending"); spiceBadge.classList.add("err");
      spiceBadge.textContent = "SPICE: error";
      spiceBadge.title = st.error;
    } else {
      spiceBadge.classList.remove("pending"); spiceBadge.classList.add("ok");
      spiceBadge.textContent = `SPICE: ${st.loaded_kernels.length} kernel${st.loaded_kernels.length === 1 ? "" : "s"}`;
      spiceBadge.title = st.loaded_kernels.length
        ? st.loaded_kernels.join("\n")
        : "no kernels loaded — 3rd-body/SRP perturbations and ephemeris-driven Moon will fail";
    }
  } catch (e) {
    spiceBadge.classList.remove("pending"); spiceBadge.classList.add("err");
    spiceBadge.textContent = "SPICE: down";
    spiceBadge.title = (e as Error).message;
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
      validateIc();
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

  // ---- Initial-state preset dropdown ----
  (document.getElementById("ic-preset") as HTMLSelectElement)
    .addEventListener("change", (e) => {
      const key = (e.target as HTMLSelectElement).value;
      const fn = PRESETS[key];
      if (!fn) return;
      fn(editorState);
      fillIc();
      validateIc();
      applyEditor(renderer).catch((err) => setStatus(`preset apply failed: ${(err as Error).message}`));
    });

  // ---- IC-form (Cartesian / Elements / ECEF / CR3BP) toggle ----
  // Every J2000-family lens (cartesian, elements, ecef) reads to and writes
  // from the same canonical `(rx..vz)` store. Switching is a round-trip:
  // read the *old* lens into canonical, then fill the *new* lens.
  (document.getElementById("ic-form") as HTMLSelectElement)
    .addEventListener("change", (e) => {
      const newForm = (e.target as HTMLSelectElement).value as IcForm;
      // Capture the current lens contents into canonical J2000 before flipping.
      if (editorState.ic_form === "elements" && newForm !== "elements") readElementsToState();
      if (editorState.ic_form === "ecef"     && newForm !== "ecef")     readEcefToState();
      editorState.ic_form = newForm;
      // Repopulate the destination lens from canonical.
      if (newForm === "elements") fillElementsFromState();
      if (newForm === "ecef")     fillEcefFromState();
      if (newForm === "cr3bp") editorState.mode = "CR3BP";
      else if (editorState.mode === "CR3BP") editorState.mode = "J2000";
      (document.getElementById("ic-mode") as HTMLSelectElement).value = editorState.mode;
      updateIcFormVisibility();
    });

  // ---- Propagation mode (J2000 / CR3BP) toggle ----
  (document.getElementById("ic-mode") as HTMLSelectElement)
    .addEventListener("change", (e) => {
      editorState.mode = (e.target as HTMLSelectElement).value as PropMode;
      if (editorState.mode === "CR3BP") {
        editorState.ic_form = "cr3bp";
        (document.getElementById("ic-form") as HTMLSelectElement).value = "cr3bp";
      }
      updateIcFormVisibility();
    });

  // ---- IC duration/maneuver inputs → revalidate + redraw event-timeline ----
  ["ic-duration", "ic-rx", "ic-ry", "ic-rz", "ic-vx", "ic-vy", "ic-vz"].forEach((id) => {
    document.getElementById(id)?.addEventListener("input", () => {
      readIc();
      validateIc();
      renderEventTimeline();
    });
  });

  // ---- Recenter camera on spacecraft ----
  btnRecenter.addEventListener("click", () => {
    if (latestCraftR) {
      camTargetSel.value = "craft";
      renderer.setCameraTarget(latestCraftR);
    }
  });

  // ---- Save / load mission JSON ----
  btnSaveMission.addEventListener("click", () => saveMissionFile());
  fileLoadMission.addEventListener("change", async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    try {
      await loadMissionFile(f);
      setStatus(`loaded mission from ${f.name}`);
      await applyEditor(renderer);
    } catch (err) {
      setStatus(`load failed: ${(err as Error).message}`);
    }
    (e.target as HTMLInputElement).value = "";   // allow reloading the same file
  });

  // ---- Trajectory export ----
  btnExportTraj.addEventListener("click", () => exportTrajectoryCsv());

  // ---- Cancel-streaming button ----
  btnCancel.addEventListener("click", () => {
    if (streamAbort) streamAbort.abort();
  });

  // ---- Footer chart kind selector ----
  chartKindSel.addEventListener("change", () => redrawFooterChart());

  // ---- Play / pause animation ----
  btnPlay.addEventListener("click", () => {
    if (playing) stopPlayback(); else startPlayback();
  });

  // ---- Manual scrub interrupts playback ----
  scrub.addEventListener("pointerdown", () => { if (playing) stopPlayback(); });

  // ---- Spacebar toggles playback (when not typing in an input). ----
  window.addEventListener("keydown", (e) => {
    if (e.code !== "Space") return;
    const tgt = e.target as HTMLElement | null;
    if (tgt && (tgt.tagName === "INPUT" || tgt.tagName === "SELECT" || tgt.tagName === "TEXTAREA")) return;
    e.preventDefault();
    if (playing) stopPlayback(); else startPlayback();
  });

  // ---- Lyapunov panel ----
  (document.getElementById("btn-lyap-run") as HTMLButtonElement).addEventListener("click", () => {
    runLyapunovPanel(renderer).catch((e) =>
      ((document.getElementById("lyap-info") as HTMLElement).textContent = `error: ${(e as Error).message}`));
  });

  // ---- WSB panel ----
  (document.getElementById("btn-wsb-run") as HTMLButtonElement).addEventListener("click", () => {
    runWsbPanel(renderer).catch((e) =>
      ((document.getElementById("wsb-info") as HTMLElement).textContent = `error: ${(e as Error).message}`));
  });

  // ---- Launch config panel ----
  (document.getElementById("btn-lc-defaults") as HTMLButtonElement).addEventListener("click", () =>
    fillLaunchDefaults().catch((e) => setStatus(`launch defaults: ${(e as Error).message}`)));
  (document.getElementById("btn-lc-run") as HTMLButtonElement).addEventListener("click", () => {
    runLaunchPanel(renderer).catch((e) =>
      ((document.getElementById("lc-info") as HTMLElement).textContent = `error: ${(e as Error).message}`));
  });

  // ---- Restore autosaved mission, if any ----
  if (restoreAutosave()) {
    fillIc();
    fillPerturbations();
    renderManeuverList();
    validateIc();
    setStatus("restored last mission from autosave");
  } else {
    fillLaunchDefaults().catch(() => {/* silent */});
  }

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
