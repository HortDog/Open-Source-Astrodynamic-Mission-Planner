export type Health = { status: string; version: string };

export async function fetchHealth(): Promise<Health> {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export type Vec3 = [number, number, number];
export type TwoBodyState = { r: Vec3; v: Vec3 };
export type VehicleSpec = {
  mass_kg: number;
  drag_area_m2?: number;
  drag_cd?: number;
  srp_area_m2?: number;
  srp_cr?: number;
};
export type ManeuverSpec = { t_offset_s: number; dv_ric: Vec3 };
export type FiniteBurnSpec = {
  t_start_s: number;
  duration_s: number;
  thrust_n: number;
  isp_s: number;
  direction_ric: Vec3;
};

export type PropagateRequest = {
  state: TwoBodyState;
  duration_s: number;
  steps?: number;
  body_name?: "EARTH" | "MOON" | "SUN";
  mu?: number;
  body_radius?: number;
  j2_enabled?: boolean;
  jn_max?: number;          // 2..6 — pass 6 for J2..J6 full set
  drag?: boolean;
  srp?: boolean;
  third_body?: string[];    // ["MOON", "SUN", ...]
  vehicle?: VehicleSpec;
  t0_tdb?: number;
  maneuvers?: ManeuverSpec[];
  finite_burns?: FiniteBurnSpec[];
  initial_mass_kg?: number;
  drag_model?: "exponential" | "msis";
  integrator?: "dop853" | "verlet" | "yoshida4";
};

export type PropagateResponse = {
  t: number[];
  states: number[][];
  perturbations?: string[];
};

export async function propagate(req: PropagateRequest): Promise<PropagateResponse> {
  const r = await fetch("/api/propagate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`propagate ${r.status}: ${await r.text()}`);
  return r.json();
}

export type HohmannResponse = {
  dv1_m_s: number;
  dv2_m_s: number;
  dv_total_m_s: number;
  transfer_time_s: number;
  semi_major_axis_m: number;
};

export async function optimizeHohmann(req: {
  r1_m: number; r2_m: number; mu?: number;
}): Promise<HohmannResponse> {
  const r = await fetch("/api/optimize/hohmann", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`hohmann ${r.status}: ${await r.text()}`);
  return r.json();
}

export type LambertResponse = {
  v1_m_s: Vec3;
  v2_m_s: Vec3;
  iterations: number;
  converged: boolean;
  transfer_time_s: number;
};

export async function optimizeLambert(req: {
  r1_m: Vec3; r2_m: Vec3; tof_s: number; mu?: number; prograde?: boolean;
}): Promise<LambertResponse> {
  const r = await fetch("/api/optimize/lambert", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`lambert ${r.status}: ${await r.text()}`);
  return r.json();
}

export type MultiBurnRequest = {
  x0_r: Vec3; x0_v: Vec3;
  xf_r: Vec3; xf_v: Vec3;
  maneuver_epochs_s: number[];
  t_final_s: number;
  mu?: number;
  initial_dv_guess?: Vec3[];
};

export type MultiBurnResponse = {
  dv_inertial_m_s: Vec3[];
  total_dv_m_s: number;
  converged: boolean;
  iterations: number;
  final_state_m: Vec3;
  final_velocity_m_s: Vec3;
};

export async function optimizeMultiBurn(req: MultiBurnRequest): Promise<MultiBurnResponse> {
  const r = await fetch("/api/optimize/multi-burn", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`multi-burn ${r.status}: ${await r.text()}`);
  return r.json();
}

export type TleResponse = {
  name: string;
  norad_id: number;
  epoch_jd: number;
  state: { r: Vec3; v: Vec3 };
  altitude_km: number;
  speed_m_s: number;
};

export async function tleByNorad(norad: number, at_utc?: string): Promise<TleResponse> {
  const url = `/api/tle/${norad}` + (at_utc ? `?at_utc=${encodeURIComponent(at_utc)}` : "");
  const r = await fetch(url);
  if (!r.ok) throw new Error(`tle ${r.status}: ${await r.text()}`);
  return r.json();
}

export async function tleParse(
  line1: string, line2: string, name = "", at_utc?: string,
): Promise<TleResponse> {
  const r = await fetch("/api/tle/parse", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ line1, line2, name, at_utc }),
  });
  if (!r.ok) throw new Error(`tle ${r.status}: ${await r.text()}`);
  return r.json();
}

export type LaunchResponse = {
  t: number[];
  states: number[][];
  burnout_index: number;
  circularization_index: number;
  burnout_time_s: number;
  circularization_dv_m_s: number;
  final_apoapsis_km: number;
  final_periapsis_km: number;
  final_speed_m_s: number;
};

export async function runLaunch(): Promise<LaunchResponse> {
  const r = await fetch("/api/launch", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "null",
  });
  if (!r.ok) throw new Error(`launch ${r.status}: ${await r.text()}`);
  return r.json();
}

export type SpiceState = {
  et: number;
  r: Vec3;
  v: Vec3;
  frame: string;
  observer: string;
};

export async function spiceState(
  target: string,
  utc: string,
  observer = "EARTH",
  frame = "J2000",
): Promise<SpiceState> {
  const r = await fetch("/api/spice/state", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ target, utc, observer, frame }),
  });
  if (!r.ok) throw new Error(`spice ${r.status}: ${await r.text()}`);
  return r.json();
}

// --------------------------------------------------------------------------- //
//  Phase 4 — CR3BP and rotating frames
// --------------------------------------------------------------------------- //

export type LagrangeResponse = {
  mu: number;
  L1: Vec3; L2: Vec3; L3: Vec3; L4: Vec3; L5: Vec3;
};

export async function cr3bpLagrange(
  opts: { mu?: number; system?: "EARTH_MOON" | "SUN_EARTH" },
): Promise<LagrangeResponse> {
  const qp = new URLSearchParams();
  if (opts.mu !== undefined) qp.set("mu", String(opts.mu));
  if (opts.system !== undefined) qp.set("system", opts.system);
  const r = await fetch(`/api/cr3bp/lagrange?${qp.toString()}`);
  if (!r.ok) throw new Error(`cr3bp ${r.status}: ${await r.text()}`);
  return r.json();
}

export type TransformStatesResponse = {
  frame: string;
  direction: "to_synodic" | "from_synodic";
  t_tdb: number;
  states: number[][];
  length_scale_m: number;
  mean_motion_rad_s: number;
};

export async function transformStates(req: {
  direction: "to_synodic" | "from_synodic";
  frame?: "EM_SYNODIC";
  t_tdb: number;
  t_offsets_s?: number[];
  states: number[][];
}): Promise<TransformStatesResponse> {
  const r = await fetch("/api/transform/states", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ frame: "EM_SYNODIC", ...req }),
  });
  if (!r.ok) throw new Error(`transform ${r.status}: ${await r.text()}`);
  return r.json();
}

export type Cr3bpPropagateResponse = {
  t: number[];
  states: number[][];
  jacobi: number[];
  mu: number;
};

export async function cr3bpPropagate(req: {
  state: [number, number, number, number, number, number];
  t_span: [number, number];
  mu?: number;
  steps?: number;
}): Promise<Cr3bpPropagateResponse> {
  const r = await fetch("/api/cr3bp/propagate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`cr3bp ${r.status}: ${await r.text()}`);
  return r.json();
}

export type Cr3bpPeriodicOrbitResponse = {
  state0: [number, number, number, number, number, number];
  period: number;
  jacobi: number;
  family: string;
  dc_iterations: number;
  dc_residual: number;
  mu: number;
};

export async function cr3bpPeriodicOrbit(req: {
  family?: "lyapunov";
  L_point?: 1 | 2;
  Ax?: number;
  mu?: number;
}): Promise<Cr3bpPeriodicOrbitResponse> {
  const r = await fetch("/api/cr3bp/periodic-orbit", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ family: "lyapunov", L_point: 1, ...req }),
  });
  if (!r.ok) throw new Error(`cr3bp ${r.status}: ${await r.text()}`);
  return r.json();
}

export type Cr3bpManifoldResponse = {
  mu: number;
  direction: "stable" | "unstable";
  branch: "+" | "-";
  trajectories: number[][][];   // [tube][sample][6]
};

export async function cr3bpManifold(req: {
  orbit_state: [number, number, number, number, number, number];
  period: number;
  mu?: number;
  direction?: "stable" | "unstable";
  branch?: "+" | "-";
  n_samples?: number;
  duration?: number;
  perturbation?: number;
  steps?: number;
}): Promise<Cr3bpManifoldResponse> {
  const r = await fetch("/api/cr3bp/manifold", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`cr3bp ${r.status}: ${await r.text()}`);
  return r.json();
}

export type Cr3bpWsbResponse = {
  mu: number;
  altitudes_m: number[];
  angles_rad: number[];
  grid: number[][];   // shape (len(altitudes), len(angles))
};

export async function cr3bpWsb(req: {
  altitudes_m: number[];
  angles_rad: number[];
  mu?: number;
  duration?: number;
  escape_radius?: number;
}): Promise<Cr3bpWsbResponse> {
  const r = await fetch("/api/cr3bp/wsb", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`cr3bp ${r.status}: ${await r.text()}`);
  return r.json();
}

export type SpiceEphemeris = {
  et0: number;
  r: Vec3[];
  v: Vec3[];
  frame: string;
  observer: string;
};

export async function spiceEphemeris(
  target: string,
  t0_utc: string,
  t_offsets_s: number[],
  observer = "EARTH",
  frame = "J2000",
): Promise<SpiceEphemeris> {
  const r = await fetch("/api/spice/ephemeris", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ target, t0_utc, t_offsets_s, observer, frame }),
  });
  if (!r.ok) throw new Error(`spice ${r.status}: ${await r.text()}`);
  return r.json();
}

export type WSMessage = Record<string, unknown>;

export class SimSocket {
  private ws: WebSocket | null = null;
  private reconnectMs = 1000;

  constructor(
    private url: string,
    private onMessage: (m: WSMessage) => void,
    private onStatus: (s: string) => void,
  ) {}

  connect(): void {
    this.onStatus("connecting…");
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.addEventListener("open", () => this.onStatus("connected"));
    ws.addEventListener("message", (ev) => {
      try {
        this.onMessage(JSON.parse(ev.data));
      } catch {
        this.onMessage({ raw: String(ev.data) });
      }
    });
    ws.addEventListener("close", () => {
      this.onStatus("disconnected — retrying");
      setTimeout(() => this.connect(), this.reconnectMs);
    });
    ws.addEventListener("error", () => ws.close());
  }

  send(msg: WSMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }
}
