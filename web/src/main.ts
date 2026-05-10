import { fetchHealth, propagate, runLaunch, SimSocket } from "./api";
import { initRenderer, type Renderer } from "./render/scene";

const status = document.getElementById("status")!;
const version = document.getElementById("version")!;
const banner = document.getElementById("banner")!;
const canvas = document.getElementById("scene") as HTMLCanvasElement;
const btnLeo = document.getElementById("btn-leo") as HTMLButtonElement;
const btnLaunch = document.getElementById("btn-launch") as HTMLButtonElement;
const btnLeoJ2 = document.getElementById("btn-leo-j2") as HTMLButtonElement;

const setStatus = (s: string) => { status.textContent = s; };
const showBanner = (msg: string) => { banner.textContent = msg; banner.classList.add("show"); };

const MU_EARTH = 3.986004418e14;
const R_EARTH = 6_378_137;
const R0 = 7_000_000;
const INCLINATION_DEG = 51.6;

function selectButton(active: HTMLButtonElement): void {
  for (const b of [btnLeo, btnLaunch, btnLeoJ2]) {
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

  // Color the three phases of the trajectory.
  // ascent (under thrust)         -> orange
  // coast (engine off, to apo)    -> yellow
  // circular orbit (post-burn)    -> cyan
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

  btnLeo.addEventListener("click", () => {
    selectButton(btnLeo);
    showLeo(renderer, false).catch((e) => setStatus(`error: ${(e as Error).message}`));
  });
  btnLeoJ2.addEventListener("click", () => {
    selectButton(btnLeoJ2);
    showLeo(renderer, true).catch((e) => setStatus(`error: ${(e as Error).message}`));
  });
  btnLaunch.addEventListener("click", () => {
    selectButton(btnLaunch);
    showLaunch(renderer).catch((e) => setStatus(`error: ${(e as Error).message}`));
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
