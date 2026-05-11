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

async function main(): Promise<void> {
  const renderer = await initRenderer(canvas);
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

  await showLeo(renderer, false);

  const sock = new SimSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
    (msg) => console.debug("ws", msg),
    (s) => console.debug("ws status:", s),
  );
  sock.connect();
}

main().catch((e) => setStatus(`fatal: ${(e as Error).message}`));
