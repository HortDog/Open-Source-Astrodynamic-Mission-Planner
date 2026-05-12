export async function fetchHealth() {
    const r = await fetch("/api/health");
    if (!r.ok)
        throw new Error(`health ${r.status}`);
    return r.json();
}
export async function propagate(req) {
    const r = await fetch("/api/propagate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
    });
    if (!r.ok)
        throw new Error(`propagate ${r.status}: ${await r.text()}`);
    return r.json();
}
/** Stream a propagation as NDJSON chunks. Yields partial trajectory slices as
 *  they complete so the caller can render the orbit incrementally. */
export async function* propagateStream(req) {
    const r = await fetch("/api/propagate/stream", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
    });
    if (!r.ok)
        throw new Error(`propagate/stream ${r.status}: ${await r.text()}`);
    const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = "";
    for (;;) {
        const { done, value } = await reader.read();
        if (done)
            break;
        buf += value;
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
            if (line.trim())
                yield JSON.parse(line);
        }
    }
    if (buf.trim())
        yield JSON.parse(buf);
}
export async function optimizeHohmann(req) {
    const r = await fetch("/api/optimize/hohmann", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
    });
    if (!r.ok)
        throw new Error(`hohmann ${r.status}: ${await r.text()}`);
    return r.json();
}
export async function optimizeLambert(req) {
    const r = await fetch("/api/optimize/lambert", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
    });
    if (!r.ok)
        throw new Error(`lambert ${r.status}: ${await r.text()}`);
    return r.json();
}
export async function optimizeMultiBurn(req) {
    const r = await fetch("/api/optimize/multi-burn", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
    });
    if (!r.ok)
        throw new Error(`multi-burn ${r.status}: ${await r.text()}`);
    return r.json();
}
export async function tleByNorad(norad, at_utc) {
    const url = `/api/tle/${norad}` + (at_utc ? `?at_utc=${encodeURIComponent(at_utc)}` : "");
    const r = await fetch(url);
    if (!r.ok)
        throw new Error(`tle ${r.status}: ${await r.text()}`);
    return r.json();
}
export async function tleParse(line1, line2, name = "", at_utc) {
    const r = await fetch("/api/tle/parse", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ line1, line2, name, at_utc }),
    });
    if (!r.ok)
        throw new Error(`tle ${r.status}: ${await r.text()}`);
    return r.json();
}
export async function runLaunch() {
    const r = await fetch("/api/launch", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "null",
    });
    if (!r.ok)
        throw new Error(`launch ${r.status}: ${await r.text()}`);
    return r.json();
}
export async function spiceState(target, utc, observer = "EARTH", frame = "J2000") {
    const r = await fetch("/api/spice/state", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ target, utc, observer, frame }),
    });
    if (!r.ok)
        throw new Error(`spice ${r.status}: ${await r.text()}`);
    return r.json();
}
export async function cr3bpLagrange(opts) {
    const qp = new URLSearchParams();
    if (opts.mu !== undefined)
        qp.set("mu", String(opts.mu));
    if (opts.system !== undefined)
        qp.set("system", opts.system);
    const r = await fetch(`/api/cr3bp/lagrange?${qp.toString()}`);
    if (!r.ok)
        throw new Error(`cr3bp ${r.status}: ${await r.text()}`);
    return r.json();
}
export async function transformStates(req) {
    const r = await fetch("/api/transform/states", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ frame: "EM_SYNODIC", ...req }),
    });
    if (!r.ok)
        throw new Error(`transform ${r.status}: ${await r.text()}`);
    return r.json();
}
export async function cr3bpPropagate(req) {
    const r = await fetch("/api/cr3bp/propagate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
    });
    if (!r.ok)
        throw new Error(`cr3bp ${r.status}: ${await r.text()}`);
    return r.json();
}
export async function cr3bpPeriodicOrbit(req) {
    const r = await fetch("/api/cr3bp/periodic-orbit", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ family: "lyapunov", L_point: 1, ...req }),
    });
    if (!r.ok)
        throw new Error(`cr3bp ${r.status}: ${await r.text()}`);
    return r.json();
}
export async function cr3bpManifold(req) {
    const r = await fetch("/api/cr3bp/manifold", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
    });
    if (!r.ok)
        throw new Error(`cr3bp ${r.status}: ${await r.text()}`);
    return r.json();
}
export async function cr3bpWsb(req) {
    const r = await fetch("/api/cr3bp/wsb", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
    });
    if (!r.ok)
        throw new Error(`cr3bp ${r.status}: ${await r.text()}`);
    return r.json();
}
export async function spiceEphemeris(target, t0_utc, t_offsets_s, observer = "EARTH", frame = "J2000") {
    const r = await fetch("/api/spice/ephemeris", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ target, t0_utc, t_offsets_s, observer, frame }),
    });
    if (!r.ok)
        throw new Error(`spice ${r.status}: ${await r.text()}`);
    return r.json();
}
export class SimSocket {
    url;
    onMessage;
    onStatus;
    ws = null;
    reconnectMs = 1000;
    constructor(url, onMessage, onStatus) {
        this.url = url;
        this.onMessage = onMessage;
        this.onStatus = onStatus;
    }
    connect() {
        this.onStatus("connecting…");
        const ws = new WebSocket(this.url);
        this.ws = ws;
        ws.addEventListener("open", () => this.onStatus("connected"));
        ws.addEventListener("message", (ev) => {
            try {
                this.onMessage(JSON.parse(ev.data));
            }
            catch {
                this.onMessage({ raw: String(ev.data) });
            }
        });
        ws.addEventListener("close", () => {
            this.onStatus("disconnected — retrying");
            setTimeout(() => this.connect(), this.reconnectMs);
        });
        ws.addEventListener("error", () => ws.close());
    }
    send(msg) {
        if (this.ws?.readyState === WebSocket.OPEN)
            this.ws.send(JSON.stringify(msg));
    }
}
