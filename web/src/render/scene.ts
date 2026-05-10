// WebGPU 3D wireframe renderer with orbit camera (mouse drag to rotate, wheel
// to zoom). Astrodynamics convention: +Z is the celestial pole.

import { lookAt, mat4, multiply, perspective } from "./mat4";

export type Renderer = {
  readonly kind: "webgpu" | "webgl2-fallback" | "none";
  /** Set the meters-per-scene-unit for all subsequent geometry. */
  setSceneScale(metersPerUnit: number): void;
  /** Draw a wireframe sphere at the origin (radius in meters). */
  setCentralBody(radiusMeters: number): void;
  /** Trajectory positions in meters, length = 3 * N (XYZ contiguous).
   *  Optional per-vertex RGB colors (length = 3 * N) for phase highlighting. */
  drawTrajectory(
    positions: Float32Array<ArrayBuffer>,
    colors?: Float32Array<ArrayBuffer>,
  ): void;
  resize(): void;
};

const SHADER = /* wgsl */ `
struct U { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: U;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec3<f32>,
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) color: vec3<f32>) -> VOut {
  var o: VOut;
  o.pos = u.mvp * vec4<f32>(pos, 1.0);
  o.color = color;
  return o;
}

@fragment
fn fs(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
  return vec4<f32>(color, 1.0);
}
`;

function nullRenderer(kind: "none" | "webgl2-fallback" = "none"): Renderer {
  return {
    kind,
    setSceneScale: () => {},
    setCentralBody: () => {},
    drawTrajectory: () => {},
    resize: () => {},
  };
}

const TRAJECTORY_COLOR: [number, number, number] = [0.55, 0.82, 1.0];
const BODY_COLOR: [number, number, number] = [0.30, 0.50, 0.70];
const BODY_EQUATOR_COLOR: [number, number, number] = [0.55, 0.75, 0.95];
const AXIS_COLORS: ReadonlyArray<[number, number, number]> = [
  [1.0, 0.35, 0.35], // +X red
  [0.4, 1.0,  0.45], // +Y green
  [0.45, 0.6, 1.0],  // +Z blue
];

// Axes drawn in meters; caller passes the desired length so they scale with
// the scene. 6 vertices = 3 line segments (origin -> +X/Y/Z).
function buildAxes(lengthMeters: number): Float32Array<ArrayBuffer> {
  const L = lengthMeters;
  const out = new Float32Array(6 * 6);
  let o = 0;
  for (let i = 0; i < 3; i++) {
    const c = AXIS_COLORS[i]!;
    out[o++] = 0; out[o++] = 0; out[o++] = 0;
    out[o++] = c[0]; out[o++] = c[1]; out[o++] = c[2];
    out[o++] = i === 0 ? L : 0;
    out[o++] = i === 1 ? L : 0;
    out[o++] = i === 2 ? L : 0;
    out[o++] = c[0]; out[o++] = c[1]; out[o++] = c[2];
  }
  return out;
}

// Wireframe sphere as a line list: parallels (latitude rings) + meridians.
// Returns Nx6 floats (xyz + rgb). Equator gets the brighter color so the
// orientation reads at a glance.
function buildSphere(
  radiusMeters: number,
  parallels = 7,
  meridians = 12,
  ringSegs = 36,
  meridianSegs = 18,
): Float32Array<ArrayBuffer> {
  const verts: number[] = [];
  const r = radiusMeters;

  for (let p = 1; p < parallels; p++) {
    const phi = -Math.PI / 2 + (p / parallels) * Math.PI;
    const isEquator = Math.abs(phi) < 1e-9;
    const c = isEquator ? BODY_EQUATOR_COLOR : BODY_COLOR;
    const cz = r * Math.sin(phi);
    const cr = r * Math.cos(phi);
    for (let i = 0; i < ringSegs; i++) {
      const t1 = (i / ringSegs) * 2 * Math.PI;
      const t2 = ((i + 1) / ringSegs) * 2 * Math.PI;
      verts.push(cr * Math.cos(t1), cr * Math.sin(t1), cz, c[0], c[1], c[2]);
      verts.push(cr * Math.cos(t2), cr * Math.sin(t2), cz, c[0], c[1], c[2]);
    }
  }

  const c = BODY_COLOR;
  for (let m = 0; m < meridians; m++) {
    const t = (m / meridians) * 2 * Math.PI;
    const ct = Math.cos(t), st = Math.sin(t);
    for (let i = 0; i < meridianSegs; i++) {
      const p1 = -Math.PI / 2 + (i / meridianSegs) * Math.PI;
      const p2 = -Math.PI / 2 + ((i + 1) / meridianSegs) * Math.PI;
      const r1 = r * Math.cos(p1), z1 = r * Math.sin(p1);
      const r2 = r * Math.cos(p2), z2 = r * Math.sin(p2);
      verts.push(r1 * ct, r1 * st, z1, c[0], c[1], c[2]);
      verts.push(r2 * ct, r2 * st, z2, c[0], c[1], c[2]);
    }
  }

  return new Float32Array(verts);
}

export async function initRenderer(canvas: HTMLCanvasElement): Promise<Renderer> {
  if (!("gpu" in navigator)) return nullRenderer();
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return nullRenderer();
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext("webgpu");
  if (!ctx) return nullRenderer();

  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "premultiplied" });

  const shader = device.createShaderModule({ code: SHADER });
  const vertexLayout: GPUVertexBufferLayout = {
    arrayStride: 24, // 3 pos + 3 color, all f32
    attributes: [
      { shaderLocation: 0, offset: 0,  format: "float32x3" },
      { shaderLocation: 1, offset: 12, format: "float32x3" },
    ],
  };

  const stripPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: shader, entryPoint: "vs", buffers: [vertexLayout] },
    fragment: { module: shader, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "line-strip" },
  });
  const listPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: shader, entryPoint: "vs", buffers: [vertexLayout] },
    fragment: { module: shader, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "line-list" },
  });

  const uniformBuf = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const stripBindGroup = device.createBindGroup({
    layout: stripPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuf } }],
  });
  const listBindGroup = device.createBindGroup({
    layout: listPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuf } }],
  });

  let axesBuf: GPUBuffer | null = null;
  function uploadAxes(): void {
    const data = buildAxes(1.1 * sceneScale);
    if (!axesBuf || axesBuf.size !== data.byteLength) {
      if (axesBuf) axesBuf.destroy();
      axesBuf = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }
    device.queue.writeBuffer(axesBuf, 0, data);
  }

  let trajBuf: GPUBuffer | null = null;
  let trajVertexCount = 0;
  let bodyBuf: GPUBuffer | null = null;
  let bodyVertexCount = 0;

  // Camera state — spherical around origin, +Z up. All distances in scene units.
  let yaw = 0.6;
  let pitch = 0.55;
  let distance = 3.0;
  let aspect = 1;

  // 1 scene unit == sceneScale meters. Set via setSceneScale().
  let sceneScale = 1;

  const proj = mat4();
  const view = mat4();
  const scaleMat = mat4();
  const viewScale = mat4();
  const mvp = mat4();

  function rebuildScaleMat(): void {
    const s = 1 / sceneScale;
    scaleMat.fill(0);
    scaleMat[0] = s; scaleMat[5] = s; scaleMat[10] = s; scaleMat[15] = 1;
  }

  function updateCamera(): void {
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cy = Math.cos(yaw),   sy = Math.sin(yaw);
    const eye: [number, number, number] = [distance * cp * sy, distance * cp * cy, distance * sp];
    perspective(proj, (50 * Math.PI) / 180, aspect, 0.01, 100);
    lookAt(view, eye, [0, 0, 0], [0, 0, 1]);
    multiply(viewScale, view, scaleMat);
    multiply(mvp, proj, viewScale);
    device.queue.writeBuffer(uniformBuf, 0, mvp);
  }

  rebuildScaleMat();

  function resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    aspect = canvas.width / canvas.height;
    updateCamera();
  }

  function frame(): void {
    const enc = device.createCommandEncoder();
    const view = ctx!.getCurrentTexture().createView();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0.03, g: 0.05, b: 0.09, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });

    pass.setPipeline(listPipeline);
    pass.setBindGroup(0, listBindGroup);

    if (bodyBuf && bodyVertexCount > 0) {
      pass.setVertexBuffer(0, bodyBuf);
      pass.draw(bodyVertexCount);
    }

    if (axesBuf) {
      pass.setVertexBuffer(0, axesBuf);
      pass.draw(6);
    }

    if (trajBuf && trajVertexCount > 0) {
      pass.setPipeline(stripPipeline);
      pass.setBindGroup(0, stripBindGroup);
      pass.setVertexBuffer(0, trajBuf);
      pass.draw(trajVertexCount);
    }

    pass.end();
    device.queue.submit([enc.finish()]);
    requestAnimationFrame(frame);
  }

  function drawTrajectory(
    positions: Float32Array<ArrayBuffer>,
    colors?: Float32Array<ArrayBuffer>,
  ): void {
    const n = positions.length / 3;
    const interleaved = new Float32Array(n * 6);
    for (let i = 0; i < n; i++) {
      interleaved[i * 6 + 0] = positions[i * 3 + 0] ?? 0;
      interleaved[i * 6 + 1] = positions[i * 3 + 1] ?? 0;
      interleaved[i * 6 + 2] = positions[i * 3 + 2] ?? 0;
      if (colors) {
        interleaved[i * 6 + 3] = colors[i * 3 + 0] ?? TRAJECTORY_COLOR[0];
        interleaved[i * 6 + 4] = colors[i * 3 + 1] ?? TRAJECTORY_COLOR[1];
        interleaved[i * 6 + 5] = colors[i * 3 + 2] ?? TRAJECTORY_COLOR[2];
      } else {
        interleaved[i * 6 + 3] = TRAJECTORY_COLOR[0];
        interleaved[i * 6 + 4] = TRAJECTORY_COLOR[1];
        interleaved[i * 6 + 5] = TRAJECTORY_COLOR[2];
      }
    }
    if (trajBuf) trajBuf.destroy();
    trajBuf = device.createBuffer({
      size: interleaved.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(trajBuf, 0, interleaved);
    trajVertexCount = n;
  }

  function setCentralBody(radiusMeters: number): void {
    const data = buildSphere(radiusMeters);
    if (bodyBuf) bodyBuf.destroy();
    bodyBuf = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bodyBuf, 0, data);
    bodyVertexCount = data.length / 6;
  }

  function setSceneScale(metersPerUnit: number): void {
    sceneScale = metersPerUnit;
    rebuildScaleMat();
    uploadAxes();
    updateCamera();
  }

  // --- mouse / wheel orbit controls ---
  let dragging = false;
  let lastX = 0, lastY = 0;
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    yaw -= dx * 0.005;
    pitch += dy * 0.005;
    const lim = Math.PI / 2 - 0.05;
    if (pitch > lim) pitch = lim;
    if (pitch < -lim) pitch = -lim;
    updateCamera();
  });
  const endDrag = (e: PointerEvent) => {
    dragging = false;
    canvas.releasePointerCapture(e.pointerId);
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    distance *= Math.exp(e.deltaY * 0.001);
    if (distance < 0.5) distance = 0.5;
    if (distance > 50) distance = 50;
    updateCamera();
  }, { passive: false });

  uploadAxes();
  resize();
  window.addEventListener("resize", resize);
  requestAnimationFrame(frame);

  return { kind: "webgpu", setSceneScale, setCentralBody, drawTrajectory, resize };
}
