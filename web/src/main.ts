import {
  fetchHealth,
  optimizeHohmann,
  optimizeLambert,
  propagate,
  runLaunch,
  SimSocket,
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
const btnEditor = document.getElementById("btn-editor") as HTMLButtonElement;
const editor = document.getElementById("editor") as HTMLElement;
const scrub = document.getElementById("scrub") as HTMLInputElement;
const scrubInfo = document.getElementById("scrub-info") as HTMLSpanElement;
const allButtons = [btnLeo, btnLaunch, btnLeoJ2, btnHohmann, btnLambert];

const setStatus = (s: string) => { status.textContent = s; };
const showBanner = (msg: string) => { banner.textContent = msg; banner.classList.add("show"); };

const MU_EARTH = 3.986004418e14;
const R_EARTH = 6_378_137;
const R0 = 7_000_000;
const INCLINATION_DEG = 51.6;

// Phase colours (phosphor palette + accents for distinct transfer phases).
const C_DEPART:  [number, number, number] = [0.20, 1.00, 0.30]; // bright phosphor — initial orbit
const C_TRANSFER:[number, number, number] = [1.00, 0.70, 0.10]; // amber — transfer arc
const C_ARRIVE:  [number, number, number] = [0.30, 0.70, 1.00]; // cyan — destination orbit

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

function setActiveTrajectory(t: number[], states: number[][]): void {
  activeTraj = { t, states };
  scrub.min = "0";
  scrub.max = String(Math.max(0, states.length - 1));
  scrub.value = "0";
  updateScrubMarker(0);
}

function updateScrubMarker(idx: number): void {
  if (!activeTraj || !activeRenderer) return;
  const s = activeTraj.states[idx];
  const t = activeTraj.t[idx];
  if (!s || t === undefined) return;
  activeRenderer.setMarker(
    [s[0]!, s[1]!, s[2]!],
    s.length >= 6 ? [s[3]!, s[4]!, s[5]!] : null,
  );
  const r = Math.hypot(s[0]!, s[1]!, s[2]!);
  const v = Math.hypot(s[3]!, s[4]!, s[5]!);
  scrubInfo.textContent =
    `t=${t.toFixed(0)} s · |r|=${(r / 1000).toFixed(0)} km · |v|=${(v / 1000).toFixed(2)} km/s`;
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

// --------------------------------------------------------------------------- //
//  Maneuver editor — sidebar panel that lets the user set IC, add Δv kicks,
//  and re-propagate the trajectory.
// --------------------------------------------------------------------------- //

type ManeuverRow = { t_offset_s: number; dv_r: number; dv_i: number; dv_c: number };

const editorState: {
  rx: number; ry: number; rz: number;
  vx: number; vy: number; vz: number;
  duration_s: number;
  steps: number;
  maneuvers: ManeuverRow[];
  j2: boolean;
  jn_max: number;
  drag: boolean;
  drag_mass_kg: number;
  drag_area_m2: number;
  drag_cd: number;
} = {
  rx: R0, ry: 0, rz: 0,
  vx: 0, vy: Math.sqrt(MU_EARTH / R0), vz: 0,
  duration_s: 2 * Math.PI * Math.sqrt((R0 * R0 * R0) / MU_EARTH),
  steps: 400,
  maneuvers: [],
  j2: false,
  jn_max: 2,
  drag: false,
  drag_mass_kg: 500,
  drag_area_m2: 4.0,
  drag_cd: 2.2,
};

function fillPerturbations(): void {
  (document.getElementById("pert-j2") as HTMLInputElement).checked = editorState.j2;
  (document.getElementById("pert-drag") as HTMLInputElement).checked = editorState.drag;
  (document.getElementById("pert-mass") as HTMLInputElement).value = String(editorState.drag_mass_kg);
  (document.getElementById("pert-area") as HTMLInputElement).value = String(editorState.drag_area_m2);
  (document.getElementById("pert-cd") as HTMLInputElement).value = String(editorState.drag_cd);
  (document.getElementById("pert-vehicle") as HTMLElement).style.display =
    editorState.drag ? "grid" : "none";
}

function readPerturbations(): void {
  const chk = (id: string) => (document.getElementById(id) as HTMLInputElement).checked;
  const n = (id: string) => parseFloat((document.getElementById(id) as HTMLInputElement).value);
  editorState.j2 = chk("pert-j2");
  editorState.drag = chk("pert-drag");
  editorState.drag_mass_kg = n("pert-mass");
  editorState.drag_area_m2 = n("pert-area");
  editorState.drag_cd = n("pert-cd");
}

function fillIc(): void {
  (document.getElementById("ic-rx") as HTMLInputElement).value = String(editorState.rx);
  (document.getElementById("ic-ry") as HTMLInputElement).value = String(editorState.ry);
  (document.getElementById("ic-rz") as HTMLInputElement).value = String(editorState.rz);
  (document.getElementById("ic-vx") as HTMLInputElement).value = String(editorState.vx);
  (document.getElementById("ic-vy") as HTMLInputElement).value = String(editorState.vy);
  (document.getElementById("ic-vz") as HTMLInputElement).value = String(editorState.vz);
  (document.getElementById("ic-duration") as HTMLInputElement).value = editorState.duration_s.toFixed(0);
  (document.getElementById("ic-steps") as HTMLInputElement).value = String(editorState.steps);
}

function readIc(): void {
  const n = (id: string) => parseFloat((document.getElementById(id) as HTMLInputElement).value);
  editorState.rx = n("ic-rx"); editorState.ry = n("ic-ry"); editorState.rz = n("ic-rz");
  editorState.vx = n("ic-vx"); editorState.vy = n("ic-vy"); editorState.vz = n("ic-vz");
  editorState.duration_s = n("ic-duration");
  editorState.steps = parseInt((document.getElementById("ic-steps") as HTMLInputElement).value, 10);
}

function renderManeuverList(): void {
  const list = document.getElementById("man-list")!;
  list.innerHTML = "";
  if (editorState.maneuvers.length === 0) {
    list.innerHTML = '<div style="opacity:.6">no maneuvers — click "+ Add" to insert one</div>';
    return;
  }
  editorState.maneuvers.forEach((m, idx) => {
    const div = document.createElement("div");
    div.className = "maneuver";
    div.innerHTML = `
      <div class="row" style="margin-bottom:3px">
        <strong>#${idx + 1}</strong>
        <label style="flex:1">t (s)
          <input data-i="${idx}" data-k="t_offset_s" type="number" step="60" value="${m.t_offset_s}" />
        </label>
        <button data-rm="${idx}" class="danger">×</button>
      </div>
      <div class="vec3">
        <input data-i="${idx}" data-k="dv_r" type="number" step="10" value="${m.dv_r}" placeholder="R" />
        <input data-i="${idx}" data-k="dv_i" type="number" step="10" value="${m.dv_i}" placeholder="I" />
        <input data-i="${idx}" data-k="dv_c" type="number" step="10" value="${m.dv_c}" placeholder="C" />
      </div>
    `;
    list.appendChild(div);
  });

  // Wire up the per-row inputs.
  list.querySelectorAll("input[data-i]").forEach((el) => {
    el.addEventListener("input", () => {
      const i = parseInt(el.getAttribute("data-i")!, 10);
      const k = el.getAttribute("data-k") as keyof ManeuverRow;
      const v = parseFloat((el as HTMLInputElement).value);
      if (Number.isFinite(v)) editorState.maneuvers[i]![k] = v;
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
  const data = await propagate({
    state: {
      r: [editorState.rx, editorState.ry, editorState.rz],
      v: [editorState.vx, editorState.vy, editorState.vz],
    },
    duration_s: editorState.duration_s,
    steps: editorState.steps,
    mu: MU_EARTH,
    body_radius: R_EARTH,
    ...(editorState.j2 ? { j2_enabled: true as const, jn_max: editorState.jn_max } : {}),
    ...(editorState.drag ? {
      drag: true as const,
      vehicle: {
        mass_kg: editorState.drag_mass_kg,
        drag_area_m2: editorState.drag_area_m2,
        drag_cd: editorState.drag_cd,
      },
    } : {}),
    maneuvers: editorState.maneuvers.map((m) => ({
      t_offset_s: m.t_offset_s,
      dv_ric: [m.dv_r, m.dv_i, m.dv_c],
    })),
  });

  // Auto-scale the scene to fit the trajectory (max radius * 1.2).
  let maxR = 0;
  for (const s of data.states) {
    const r = Math.hypot(s[0]!, s[1]!, s[2]!);
    if (r > maxR) maxR = r;
  }
  renderer.setSceneScale(Math.max(R0, maxR * 1.2));
  renderer.setCentralBody(R_EARTH);

  // Phase colours by manoeuvre boundaries.
  const phaseTimes = [0, ...editorState.maneuvers.map((m) => m.t_offset_s)];
  const phaseColors = phaseTimes.map((_, i) => {
    if (i === 0) return C_DEPART;
    if (i === phaseTimes.length - 1) return C_ARRIVE;
    return C_TRANSFER;
  });
  const colors = phaseColors.length > 1
    ? colorByPhases(data.t, phaseTimes, phaseColors)
    : undefined;
  renderer.drawTrajectory(statesToPositions(data.states), colors);
  setActiveTrajectory(data.t, data.states);

  // Summary.
  const totalDv = editorState.maneuvers
    .reduce((s, m) => s + Math.hypot(m.dv_r, m.dv_i, m.dv_c), 0);
  const summary = document.getElementById("editor-summary")!;
  summary.innerHTML =
    `samples: ${data.states.length}<br>` +
    `peak |r|: ${(maxR / 1000).toFixed(0)} km<br>` +
    `Σ |Δv|: ${(totalDv / 1000).toFixed(3)} km/s`;
  setStatus(
    `Editor: ${editorState.maneuvers.length} maneuver(s), Σ Δv=${(totalDv / 1000).toFixed(2)} km/s`,
  );
}

function resetEditor(): void {
  editorState.rx = R0; editorState.ry = 0; editorState.rz = 0;
  editorState.vx = 0; editorState.vy = Math.sqrt(MU_EARTH / R0); editorState.vz = 0;
  editorState.duration_s = 2 * Math.PI * Math.sqrt((R0 * R0 * R0) / MU_EARTH);
  editorState.steps = 400;
  editorState.maneuvers = [];
  editorState.j2 = false; editorState.jn_max = 2;
  editorState.drag = false; editorState.drag_mass_kg = 500;
  editorState.drag_area_m2 = 4.0; editorState.drag_cd = 2.2;
  fillIc();
  fillPerturbations();
  renderManeuverList();
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
      editorState.maneuvers.push({ t_offset_s: next_t, dv_r: 0, dv_i: 50, dv_c: 0 });
      renderManeuverList();
    });
  (document.getElementById("btn-apply") as HTMLButtonElement)
    .addEventListener("click", () => {
      applyEditor(renderer).catch((e) => setStatus(`editor error: ${(e as Error).message}`));
    });
  (document.getElementById("btn-reset") as HTMLButtonElement)
    .addEventListener("click", () => resetEditor());

  (document.getElementById("pert-drag") as HTMLInputElement)
    .addEventListener("change", (e) => {
      (document.getElementById("pert-vehicle") as HTMLElement).style.display =
        (e.target as HTMLInputElement).checked ? "grid" : "none";
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
