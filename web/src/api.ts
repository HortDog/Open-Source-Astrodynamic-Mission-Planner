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
