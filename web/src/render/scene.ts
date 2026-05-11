// WebGPU 3D renderer with orbit camera. All lines are rendered as
// screen-space quads so line width is controllable. Occluded lines
// (behind the central body) appear as dotted. +Z is the celestial pole.

import { lookAt, mat4, multiply, perspective } from "./mat4";

export type Renderer = {
  readonly kind: "webgpu" | "webgl2-fallback" | "none";
  setSceneScale(metersPerUnit: number): void;
  setCentralBody(radiusMeters: number): void;
  drawTrajectory(
    positions: Float32Array<ArrayBuffer>,
    colors?: Float32Array<ArrayBuffer>,
  ): void;
  /** Place a spacecraft marker at the given inertial position (m). When a
   *  velocity (m/s) is supplied the three arms align with the RIC frame
   *  (red=radial, green=in-track, cyan=cross-track). Pass null to hide.
   *  Size auto-scales to ~3% of the current scene scale. */
  setMarker(
    position: [number, number, number] | null,
    velocity?: [number, number, number] | null,
    color?: [number, number, number],
  ): void;
  /** Render an off-centre body (e.g. the Moon at its inertial position) as a
   *  small wireframe sphere.  Pass null position to hide. */
  setSecondaryBody(
    position: [number, number, number] | null,
    radius: number,
    color?: [number, number, number],
  ): void;
  resize(): void;
};

const DEPTH_FORMAT = "depth24plus" as const;
const LINE_WIDTH = 6; // pixels

// Uniform layout (80 bytes, 16-byte aligned):
//   mvp      : mat4x4<f32>  offset  0
//   viewport : vec2<f32>    offset 64
//   lineWidth: f32           offset 72
//   _pad     : f32           offset 76
const UNIFORM_FLOATS = 20;

const SHADER = /* wgsl */ `
struct U {
  mvp:       mat4x4<f32>,
  viewport:  vec2<f32>,
  lineWidth: f32,
  _pad:      f32,
};
@group(0) @binding(0) var<uniform> u: U;

struct VOut {
  @builtin(position) pos:   vec4<f32>,
  @location(0)       color: vec3<f32>,
};

// Depth pre-pass: position only, dummy color.
@vertex
fn vs_depth(@location(0) pos: vec3<f32>) -> VOut {
  var o: VOut;
  o.pos   = u.mvp * vec4<f32>(pos, 1.0);
  o.color = vec3<f32>(0.0);
  return o;
}

// Thick-line vertex shader. Each segment (pos_a → pos_b) is expanded into a
// screen-space quad. corner.x selects the anchor end (0 = start, 1 = finish);
// corner.y is the perpendicular side (−1 or +1).
@vertex
fn vs_thick(
  @location(0) pos_a:  vec3<f32>,
  @location(1) pos_b:  vec3<f32>,
  @location(2) color:  vec3<f32>,
  @location(3) corner: vec2<f32>,
) -> VOut {
  let clip_a = u.mvp * vec4<f32>(pos_a, 1.0);
  let clip_b = u.mvp * vec4<f32>(pos_b, 1.0);
  let ndc_a  = clip_a.xy / clip_a.w;
  let ndc_b  = clip_b.xy / clip_b.w;

  // Segment direction in pixel space, normalised.
  var d = (ndc_b - ndc_a) * u.viewport * 0.5;
  let dlen = length(d);
  if (dlen > 0.001) { d = d / dlen; } else { d = vec2<f32>(1.0, 0.0); }

  // Screen-space perpendicular converted back to NDC half-width.
  let perp_ndc = vec2<f32>(-d.y, d.x) / (u.viewport * 0.5) * (u.lineWidth * 0.5);

  let base   = mix(clip_a, clip_b, corner.x);
  let offset = perp_ndc * corner.y * base.w;  // NDC → clip space

  var o: VOut;
  o.pos   = vec4<f32>(base.xy + offset, base.z, base.w);
  o.color = color;
  return o;
}

@fragment
fn fs(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
  return vec4<f32>(color, 1.0);
}

// Circular dots every 6 px for occluded lines (radius ≈ 1.2 px, dimmed).
@fragment
fn fs_dot(
  @location(0)       color: vec3<f32>,
  @builtin(position) pos:   vec4<f32>,
) -> @location(0) vec4<f32> {
  let period = 6.0;
  let cx = pos.x % period - period * 0.5;
  let cy = pos.y % period - period * 0.5;
  if (cx * cx + cy * cy > 1.5) { discard; }
  return vec4<f32>(color * 0.5, 1.0);
}
`;

function nullRenderer(kind: "none" | "webgl2-fallback" = "none"): Renderer {
  return {
    kind,
    setSceneScale:    () => {},
    setCentralBody:   () => {},
    drawTrajectory:   () => {},
    setMarker:        () => {},
    setSecondaryBody: () => {},
    resize:           () => {},
  };
}

const TRAJ_COLOR:   [number, number, number] = [0.18, 1.00, 0.08]; // bright phosphor
// Central body rendered in gray
const BODY_COLOR:   [number, number, number] = [0.50, 0.50, 0.50]; // gray
const BODY_EQ_CLR:  [number, number, number] = [0.60, 0.60, 0.60]; // equator highlight (lighter gray)
const AXIS_COLORS: ReadonlyArray<[number, number, number]> = [
  [1.00, 0.06, 0.02], // +X  red
  [0.15, 1.00, 0.10], // +Y  green
  [0.00, 0.70, 1.00], // +Z  cyan
];

// Six corner descriptors for one quad (2 triangles, CCW).
const CORNERS: ReadonlyArray<[number, number]> = [
  [0, -1], [0, +1], [1, -1],
  [1, -1], [0, +1], [1, +1],
];
const THICK_STRIDE = 11; // pos_a(3) + pos_b(3) + color(3) + corner(2)

// Convert a line-strip (N positions + optional N colors) into thick-quad vertices.
function stripToQuads(
  pos: Float32Array<ArrayBuffer>,
  cols: Float32Array<ArrayBuffer> | undefined,
  defCol: [number, number, number],
): Float32Array<ArrayBuffer> {
  const n    = pos.length / 3;
  const segs = Math.max(0, n - 1);
  const out  = new Float32Array(segs * 6 * THICK_STRIDE);
  let o = 0;
  for (let s = 0; s < segs; s++) {
    const ia = s, ib = s + 1;
    const ax = pos[ia * 3]!,  ay = pos[ia * 3 + 1]!,  az = pos[ia * 3 + 2]!;
    const bx = pos[ib * 3]!,  by = pos[ib * 3 + 1]!,  bz = pos[ib * 3 + 2]!;
    const ca: [number, number, number] = cols
      ? [cols[ia * 3]!, cols[ia * 3 + 1]!, cols[ia * 3 + 2]!]
      : defCol;
    const cb: [number, number, number] = cols
      ? [cols[ib * 3]!, cols[ib * 3 + 1]!, cols[ib * 3 + 2]!]
      : defCol;
    for (const [atEnd, side] of CORNERS) {
      out[o++] = ax;  out[o++] = ay;  out[o++] = az;
      out[o++] = bx;  out[o++] = by;  out[o++] = bz;
      const c = atEnd === 0 ? ca : cb;
      out[o++] = c[0]; out[o++] = c[1]; out[o++] = c[2];
      out[o++] = atEnd; out[o++] = side;
    }
  }
  return out;
}

// Convert a line-list (interleaved pos+color pairs, stride 6) into thick-quad vertices.
function listToQuads(data: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> {
  const segs = data.length / 12; // 2 vertices × 6 floats per pair
  const out  = new Float32Array(segs * 6 * THICK_STRIDE);
  let o = 0;
  for (let s = 0; s < segs; s++) {
    const b = s * 12;
    const ax = data[b]!,     ay = data[b+1]!,  az = data[b+2]!;
    const ar = data[b+3]!,   ag = data[b+4]!,  ab = data[b+5]!;
    const bx = data[b+6]!,   by = data[b+7]!,  bz = data[b+8]!;
    const br = data[b+9]!,   bg = data[b+10]!, bb = data[b+11]!;
    for (const [atEnd, side] of CORNERS) {
      out[o++] = ax; out[o++] = ay; out[o++] = az;
      out[o++] = bx; out[o++] = by; out[o++] = bz;
      if (atEnd === 0) { out[o++] = ar; out[o++] = ag; out[o++] = ab; }
      else             { out[o++] = br; out[o++] = bg; out[o++] = bb; }
      out[o++] = atEnd; out[o++] = side;
    }
  }
  return out;
}

// Wireframe sphere: parallels + meridians as a line-list (interleaved pos+color).
function buildSphere(
  r: number,
  parallels = 18,
  meridians = 36,
  ringSegs = 72*2,
  meridianSegs = 36*2,
): Float32Array<ArrayBuffer> {
  const verts: number[] = [];
  for (let p = 1; p < parallels; p++) {
    const phi = -Math.PI / 2 + (p / parallels) * Math.PI;
    const c   = Math.abs(phi) < 1e-9 ? BODY_EQ_CLR : BODY_COLOR;
    const cz  = r * Math.sin(phi), cr = r * Math.cos(phi);
    for (let i = 0; i < ringSegs; i++) {
      const t1 = (i / ringSegs) * 2 * Math.PI;
      const t2 = ((i + 1) / ringSegs) * 2 * Math.PI;
      verts.push(cr * Math.cos(t1), cr * Math.sin(t1), cz, c[0], c[1], c[2]);
      verts.push(cr * Math.cos(t2), cr * Math.sin(t2), cz, c[0], c[1], c[2]);
    }
  }
  const c = BODY_COLOR;
  for (let m = 0; m < meridians; m++) {
    const t  = (m / meridians) * 2 * Math.PI;
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

// Axes as a line-list (interleaved pos+color).
function buildAxes(len: number): Float32Array<ArrayBuffer> {
  const out = new Float32Array(6 * 6);
  let o = 0;
  for (let i = 0; i < 3; i++) {
    const c = AXIS_COLORS[i]!;
    out[o++] = 0; out[o++] = 0; out[o++] = 0;
    out[o++] = c[0]; out[o++] = c[1]; out[o++] = c[2];
    out[o++] = i === 0 ? len : 0;
    out[o++] = i === 1 ? len : 0;
    out[o++] = i === 2 ? len : 0;
    out[o++] = c[0]; out[o++] = c[1]; out[o++] = c[2];
  }
  return out;
}

// XYZ-only triangle mesh for the depth pre-pass (solid sphere).
function buildSolidSphere(radius: number, rings = 32, segs = 32): Float32Array<ArrayBuffer> {
  const verts: number[] = [];
  for (let ri = 0; ri < rings; ri++) {
    const phi1 = -Math.PI / 2 + (ri / rings) * Math.PI;
    const phi2 = -Math.PI / 2 + ((ri + 1) / rings) * Math.PI;
    for (let si = 0; si < segs; si++) {
      const th1 = (si / segs) * 2 * Math.PI;
      const th2 = ((si + 1) / segs) * 2 * Math.PI;
      const p = (phi: number, th: number): [number, number, number] => [
        radius * Math.cos(phi) * Math.cos(th),
        radius * Math.cos(phi) * Math.sin(th),
        radius * Math.sin(phi),
      ];
      const [p00, p10, p01, p11] = [p(phi1, th1), p(phi1, th2), p(phi2, th1), p(phi2, th2)];
      verts.push(...p00, ...p01, ...p10);
      verts.push(...p10, ...p01, ...p11);
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

  const depthLayout: GPUVertexBufferLayout = {
    arrayStride: 12,
    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
  };
  const thickLayout: GPUVertexBufferLayout = {
    arrayStride: THICK_STRIDE * 4,
    attributes: [
      { shaderLocation: 0, offset: 0,  format: "float32x3" }, // pos_a
      { shaderLocation: 1, offset: 12, format: "float32x3" }, // pos_b
      { shaderLocation: 2, offset: 24, format: "float32x3" }, // color
      { shaderLocation: 3, offset: 36, format: "float32x2" }, // corner
    ],
  };

  const ds = (write: boolean, compare: GPUCompareFunction): GPUDepthStencilState => ({
    format: DEPTH_FORMAT, depthWriteEnabled: write, depthCompare: compare,
  });

  const depthFillPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: shader, entryPoint: "vs_depth", buffers: [depthLayout] },
    fragment: { module: shader, entryPoint: "fs", targets: [{ format, writeMask: 0 }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: ds(true, "less"),
  });

  const mkThickPipeline = (fsEntry: string, compare: GPUCompareFunction) =>
    device.createRenderPipeline({
      layout: "auto",
      vertex: { module: shader, entryPoint: "vs_thick", buffers: [thickLayout] },
      fragment: { module: shader, entryPoint: fsEntry, targets: [{ format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: ds(false, compare),
    });

  const thickLePipeline     = mkThickPipeline("fs",     "less-equal"); // visible, solid
  const thickGtPipeline     = mkThickPipeline("fs_dot", "greater");    // occluded, dotted
  const thickAlwaysPipeline = mkThickPipeline("fs",     "always");     // axes on top

  const uniformBuf = device.createBuffer({
    size: UNIFORM_FLOATS * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const mkBG = (p: GPURenderPipeline): GPUBindGroup =>
    device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuf } }],
    });

  const depthFillBG     = mkBG(depthFillPipeline);
  const thickLeBG       = mkBG(thickLePipeline);
  const thickGtBG       = mkBG(thickGtPipeline);
  const thickAlwaysBG   = mkBG(thickAlwaysPipeline);

  let trajBuf:        GPUBuffer | null = null;
  let trajCount       = 0;
  let bodyBuf:        GPUBuffer | null = null;
  let bodyCount       = 0;
  let solidBodyBuf:   GPUBuffer | null = null;
  let solidBodyCount  = 0;
  let secBodyBuf:     GPUBuffer | null = null;
  let secBodyCount    = 0;
  let axesBuf:        GPUBuffer | null = null;
  let axesCount       = 0;
  let markerBuf:      GPUBuffer | null = null;
  let markerCount     = 0;

  let depthTexture: GPUTexture     | null = null;
  let depthView:    GPUTextureView | null = null;

  let yaw = 0.6, pitch = 0.55, distance = 3.0, aspect = 1;
  let sceneScale = 1;

  const proj = mat4(), view = mat4(), scaleMat = mat4(), viewScale = mat4(), mvp = mat4();
  const uData = new Float32Array(UNIFORM_FLOATS);

  function rebuildScaleMat(): void {
    const s = 1 / sceneScale;
    scaleMat.fill(0);
    scaleMat[0] = s; scaleMat[5] = s; scaleMat[10] = s; scaleMat[15] = 1;
  }

  function writeUniforms(): void {
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cy = Math.cos(yaw),   sy = Math.sin(yaw);
    const eye: [number, number, number] = [distance * cp * sy, distance * cp * cy, distance * sp];
    perspective(proj, (50 * Math.PI) / 180, aspect, 0.01, 100);
    lookAt(view, eye, [0, 0, 0], [0, 0, 1]);
    multiply(viewScale, view, scaleMat);
    multiply(mvp, proj, viewScale);
    uData.set(mvp, 0);
    uData[16] = canvas.width;
    uData[17] = canvas.height;
    uData[18] = LINE_WIDTH;
    uData[19] = 0;
    device.queue.writeBuffer(uniformBuf, 0, uData);
  }

  rebuildScaleMat();

  function uploadThick(
    buf: GPUBuffer | null,
    data: Float32Array<ArrayBuffer>,
  ): [GPUBuffer, number] {
    buf?.destroy();
    const newBuf = device.createBuffer({
      size: Math.max(data.byteLength, 4),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    if (data.length > 0) device.queue.writeBuffer(newBuf, 0, data);
    return [newBuf, data.length / THICK_STRIDE];
  }

  function uploadRaw(
    buf: GPUBuffer | null,
    data: Float32Array<ArrayBuffer>,
    countDivisor: number,
  ): [GPUBuffer, number] {
    buf?.destroy();
    const newBuf = device.createBuffer({
      size: Math.max(data.byteLength, 4),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    if (data.length > 0) device.queue.writeBuffer(newBuf, 0, data);
    return [newBuf, data.length / countDivisor];
  }

  function resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width  = Math.max(1, Math.floor(canvas.clientWidth  * dpr));
    canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    aspect = canvas.width / canvas.height;

    depthTexture?.destroy();
    depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    depthView = depthTexture.createView();

    writeUniforms();
  }

  function frame(): void {
    if (!depthView) { requestAnimationFrame(frame); return; }

    const enc  = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: ctx!.getCurrentTexture().createView(),
        clearValue: { r: 0.01, g: 0.04, b: 0.01, a: 1 },
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1.0,
        depthLoadOp: "clear", depthStoreOp: "store",
      },
    });

    // 1. Depth pre-pass: fill solid sphere into depth buffer, no colour write.
    if (solidBodyBuf && solidBodyCount > 0) {
      pass.setPipeline(depthFillPipeline);
      pass.setBindGroup(0, depthFillBG);
      pass.setVertexBuffer(0, solidBodyBuf);
      pass.draw(solidBodyCount);
    }

    // 2. Planet wireframe occluded → dotted (drawn before trajectory).
    pass.setPipeline(thickGtPipeline);
    pass.setBindGroup(0, thickGtBG);
    if (bodyBuf && bodyCount > 0) { pass.setVertexBuffer(0, bodyBuf); pass.draw(bodyCount); }

    // 3. Planet wireframe visible → solid.
    pass.setPipeline(thickLePipeline);
    pass.setBindGroup(0, thickLeBG);
    if (bodyBuf && bodyCount > 0) { pass.setVertexBuffer(0, bodyBuf); pass.draw(bodyCount); }

    // 4. Trajectory occluded → dotted (on top of planet wireframe).
    pass.setPipeline(thickGtPipeline);
    pass.setBindGroup(0, thickGtBG);
    if (trajBuf && trajCount > 0) { pass.setVertexBuffer(0, trajBuf); pass.draw(trajCount); }

    // 5. Trajectory visible → solid.
    pass.setPipeline(thickLePipeline);
    pass.setBindGroup(0, thickLeBG);
    if (trajBuf && trajCount > 0) { pass.setVertexBuffer(0, trajBuf); pass.draw(trajCount); }

    // 5b. Secondary body (Moon at its inertial position) — always visible.
    if (secBodyBuf && secBodyCount > 0) {
      pass.setPipeline(thickAlwaysPipeline);
      pass.setBindGroup(0, thickAlwaysBG);
      pass.setVertexBuffer(0, secBodyBuf);
      pass.draw(secBodyCount);
    }

    // 4. Axes: always on top.
    if (axesBuf && axesCount > 0) {
      pass.setPipeline(thickAlwaysPipeline);
      pass.setBindGroup(0, thickAlwaysBG);
      pass.setVertexBuffer(0, axesBuf);
      pass.draw(axesCount);
    }

    // 5. Spacecraft marker: always on top.
    if (markerBuf && markerCount > 0) {
      pass.setPipeline(thickAlwaysPipeline);
      pass.setBindGroup(0, thickAlwaysBG);
      pass.setVertexBuffer(0, markerBuf);
      pass.draw(markerCount);
    }

    pass.end();
    device.queue.submit([enc.finish()]);
    requestAnimationFrame(frame);
  }

  function drawTrajectory(
    positions: Float32Array<ArrayBuffer>,
    colors?: Float32Array<ArrayBuffer>,
  ): void {
    const quads = stripToQuads(positions, colors, TRAJ_COLOR);
    [trajBuf, trajCount] = uploadThick(trajBuf, quads);
  }

  function setCentralBody(radiusMeters: number): void {
    const wireQuads = listToQuads(buildSphere(radiusMeters));
    [bodyBuf, bodyCount] = uploadThick(bodyBuf, wireQuads);

    const solidData = buildSolidSphere(radiusMeters * 0.995); // slightly smaller to avoid z-fighting with wireframe
    [solidBodyBuf, solidBodyCount] = uploadRaw(solidBodyBuf, solidData, 3);
  }

  const SEC_BODY_COLOR: [number, number, number] = [0.55, 0.55, 0.55]; // moon-grey

  function setSecondaryBody(
    position: [number, number, number] | null,
    radius: number,
    color?: [number, number, number],
  ): void {
    if (position === null || radius <= 0) {
      secBodyBuf?.destroy();
      secBodyBuf = null; secBodyCount = 0;
      return;
    }
    // Build a sparse wireframe sphere translated to `position`.
    const wire = buildSphere(radius, 9, 12, 36, 18);  // lower resolution for the secondary body
    const out = new Float32Array(wire.length);
    const c = color ?? SEC_BODY_COLOR;
    for (let i = 0; i < wire.length; i += 6) {
      out[i]     = wire[i]!     + position[0];
      out[i + 1] = wire[i + 1]! + position[1];
      out[i + 2] = wire[i + 2]! + position[2];
      out[i + 3] = c[0];
      out[i + 4] = c[1];
      out[i + 5] = c[2];
    }
    const quads = listToQuads(out);
    [secBodyBuf, secBodyCount] = uploadThick(secBodyBuf, quads);
  }

  function setSceneScale(metersPerUnit: number): void {
    sceneScale = metersPerUnit;
    rebuildScaleMat();

    const axesData = listToQuads(buildAxes(0.25 * sceneScale));
    [axesBuf, axesCount] = uploadThick(axesBuf, axesData);

    writeUniforms();
  }

  // RIC arm colours (match axis palette so they read clearly).
  const MKR_R: [number, number, number] = [1.00, 0.35, 0.20]; // radial     — red/orange
  const MKR_I: [number, number, number] = [0.20, 1.00, 0.30]; // in-track   — green
  const MKR_C: [number, number, number] = [0.20, 0.80, 1.00]; // cross-track — cyan
  const MKR_DEF: [number, number, number] = [1.0, 0.95, 0.25]; // fallback — yellow

  function setMarker(
    position: [number, number, number] | null,
    velocity?: [number, number, number] | null,
    color?: [number, number, number],
  ): void {
    if (position === null) {
      markerBuf?.destroy();
      markerBuf = null;
      markerCount = 0;
      return;
    }
    const size = 0.03 * sceneScale;
    const [px, py, pz] = position;

    // Default: inertial ±X/Y/Z cross with a single colour.
    let r_hat: [number, number, number] = [1, 0, 0];
    let i_hat: [number, number, number] = [0, 1, 0];
    let c_hat: [number, number, number] = [0, 0, 1];
    let cr = color ?? MKR_DEF;
    let ci = color ?? MKR_DEF;
    let cc = color ?? MKR_DEF;

    if (velocity != null) {
      const [vx, vy, vz] = velocity;
      const rMag = Math.hypot(px, py, pz);
      if (rMag > 0) {
        r_hat = [px / rMag, py / rMag, pz / rMag];
        // Ĉ = r × v / |r × v|
        const hx = py * vz - pz * vy;
        const hy = pz * vx - px * vz;
        const hz = px * vy - py * vx;
        const hMag = Math.hypot(hx, hy, hz);
        if (hMag > 0) {
          c_hat = [hx / hMag, hy / hMag, hz / hMag];
          // Î = Ĉ × R̂
          i_hat = [
            c_hat[1] * r_hat[2] - c_hat[2] * r_hat[1],
            c_hat[2] * r_hat[0] - c_hat[0] * r_hat[2],
            c_hat[0] * r_hat[1] - c_hat[1] * r_hat[0],
          ];
          cr = MKR_R; ci = MKR_I; cc = MKR_C;
        }
      }
    }

    const lineList = new Float32Array([
      px, py, pz,  cr[0], cr[1], cr[2],
      px + r_hat[0] * size, py + r_hat[1] * size, pz + r_hat[2] * size,  cr[0], cr[1], cr[2],
      px, py, pz,  ci[0], ci[1], ci[2],
      px + i_hat[0] * size, py + i_hat[1] * size, pz + i_hat[2] * size,  ci[0], ci[1], ci[2],
      px, py, pz,  cc[0], cc[1], cc[2],
      px + c_hat[0] * size, py + c_hat[1] * size, pz + c_hat[2] * size,  cc[0], cc[1], cc[2],
    ]);
    const quads = listToQuads(lineList);
    [markerBuf, markerCount] = uploadThick(markerBuf, quads);
  }

  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    yaw   -= (e.clientX - lastX) * 0.005;
    pitch += (e.clientY - lastY) * 0.005;
    lastX = e.clientX; lastY = e.clientY;
    const lim = Math.PI / 2 - 0.05;
    pitch = Math.max(-lim, Math.min(lim, pitch));
    writeUniforms();
  });
  const endDrag = (e: PointerEvent) => {
    dragging = false; canvas.releasePointerCapture(e.pointerId);
  };
  canvas.addEventListener("pointerup",     endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    distance = Math.max(0.5, Math.min(50, distance * Math.exp(e.deltaY * 0.001)));
    writeUniforms();
  }, { passive: false });

  resize();
  window.addEventListener("resize", resize);
  requestAnimationFrame(frame);

  return { kind: "webgpu", setSceneScale, setCentralBody, drawTrajectory, setMarker, setSecondaryBody, resize };
}
