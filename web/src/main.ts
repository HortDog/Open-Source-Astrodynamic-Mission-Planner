import { fetchHealth, propagate, SimSocket } from "./api";
import { initRenderer, type Renderer } from "./render/scene";

const status = document.getElementById("status")!;
const version = document.getElementById("version")!;
const canvas = document.getElementById("scene") as HTMLCanvasElement;

const setStatus = (s: string) => { status.textContent = s; };

const MU_EARTH = 3.986004418e14;
const R_EARTH = 6_378_137; // WGS-84 equatorial radius, m
const R0 = 7_000_000;
const INCLINATION_DEG = 51.6; // ISS-like

async function loadLeoTest(renderer: Renderer): Promise<void> {
  if (renderer.kind !== "webgpu") return;
  renderer.setSceneScale(R0);
  renderer.setCentralBody(R_EARTH);

  const v0 = Math.sqrt(MU_EARTH / R0);
  const period = 2 * Math.PI * Math.sqrt((R0 * R0 * R0) / MU_EARTH);
  const i = (INCLINATION_DEG * Math.PI) / 180;

  setStatus(`propagating LEO (r=${R0 / 1000} km, i=${INCLINATION_DEG}°, T=${period.toFixed(0)} s)…`);
  const data = await propagate({
    state: { r: [R0, 0, 0], v: [0, v0 * Math.cos(i), v0 * Math.sin(i)] },
    duration_s: period,
    steps: 400,
    mu: MU_EARTH,
  });

  const positions = new Float32Array(data.states.length * 3);
  for (let i = 0; i < data.states.length; i++) {
    const s = data.states[i]!;
    positions[i * 3 + 0] = s[0]!;
    positions[i * 3 + 1] = s[1]!;
    positions[i * 3 + 2] = s[2]!;
  }
  renderer.drawTrajectory(positions);
  setStatus(`LEO orbit rendered — i=${INCLINATION_DEG}°, ${data.states.length} samples over one period (${period.toFixed(0)} s) · drag to rotate, wheel to zoom`);
}

async function main(): Promise<void> {
  const renderer = await initRenderer(canvas);
  if (renderer.kind === "none") {
    setStatus("WebGPU unavailable — fallback renderer not yet implemented");
  }

  try {
    const h = await fetchHealth();
    version.textContent = `v${h.version}`;
  } catch (e) {
    setStatus(`backend unreachable: ${(e as Error).message}`);
    return;
  }

  try {
    await loadLeoTest(renderer);
  } catch (e) {
    setStatus(`LEO test failed: ${(e as Error).message}`);
  }

  const sock = new SimSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
    (msg) => console.debug("ws", msg),
    (s) => console.debug("ws status:", s),
  );
  sock.connect();
}

main().catch((e) => setStatus(`fatal: ${(e as Error).message}`));
