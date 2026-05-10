export type Health = { status: string; version: string };

export async function fetchHealth(): Promise<Health> {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export type Vec3 = [number, number, number];
export type TwoBodyState = { r: Vec3; v: Vec3 };
export type PropagateResponse = { t: number[]; states: number[][] };

export async function propagate(req: {
  state: TwoBodyState;
  duration_s: number;
  steps?: number;
  mu?: number;
}): Promise<PropagateResponse> {
  const r = await fetch("/api/propagate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`propagate ${r.status}: ${await r.text()}`);
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
