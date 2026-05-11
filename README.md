# Open Source Astrodynamic Mission Planning (OAMP)

Interactive platform for space mission design with hybrid Newtonian / PN / GR
dynamics, SPICE-fidelity ephemerides, and a WebGPU sandbox frontend.

See the [Project Brief](Open_Source_Astrodynamics_Mission_Planning.pdf) and
[architecture diagram](docs/architecture.d2).

## Layout

- `backend/` — Python package `oamp` (FastAPI + dynamics engines)
  - `oamp.api`        — FastAPI app, REST + WebSocket
  - `oamp.bodies`     — central-body constants (mu, radius, J2)
  - `oamp.dynamics`   — `newtonian` (with J2), `launch` (gravity-turn + circ.)
  - `oamp.spice`      — kernel manifest + downloader + SI-unit query wrapper
  - `oamp.timescales` — UTC ↔ TDB ET conversions (SPICE-aware, fallback table)
- `web/` — Vite + TypeScript frontend (WebGPU)
- `docs/` — D2 architecture diagrams
- `data/kernels/` — SPICE kernels (gitignored, fetched via `pixi run kernels`)
- `pixi.toml` — Python environment and tasks

## Quick start

```sh
pixi install
pixi run test           # 10 tests: 2-body, J2 nodal regression, launch, time scales
pixi run dev            # http://localhost:8000  (FastAPI + /docs)
pixi run kernels        # download DE440s + LSK + PCK to data/kernels/
pixi run arch           # render docs/architecture.svg
```

```sh
cd web && npm install
npm run dev             # http://localhost:5173
```

## What's in the MVP

**Phase 1:**
- Newtonian propagator with optional J2 oblateness, DOP853 integrator
- Open-loop launch simulator (atmosphere drag, pitch program, apoapsis circularization)
- SPICE wrapper with reproducibility-locked kernel manifest
- WebGPU 3D viewer: orbit camera, central-body wireframe, axes, phase-colored trajectories
- Three demo scenarios in the UI: LEO orbit · Launch demo · LEO + J2 drift

**Phase 2 (this branch — Precision Newtonian + Mission Design):**

- Pluggable perturbation engine ([`oamp.dynamics.perturbations`](backend/oamp/dynamics/perturbations.py)):
  zonal harmonics J2–J6, third-body gravity (Moon/Sun via SPICE), cannonball SRP
  with cylindrical Earth shadow, exponential / piecewise drag — plus optional
  NRLMSISE / MSIS-2 density via [`oamp.dynamics.atmosphere`](backend/oamp/dynamics/atmosphere.py) (`pymsis`)
- Symplectic integrators ([`oamp.dynamics.integrators`](backend/oamp/dynamics/integrators.py)):
  velocity-Verlet (2nd order) and Yoshida-4 (4th order) for long-arc energy-preserving
  runs; selectable via `integrator: "dop853" | "verlet" | "yoshida4"` on `/propagate`
- Impulsive Δv manoeuvres in RIC frame, applied between integration arcs
- Lambert universal-variable solver (Curtis 5.3, scipy `brentq` root-finder)
- Closed-form Hohmann transfer
- Fuel-optimal multi-burn NLP via CasADi + IPOPT
  ([`oamp.dynamics.optimization`](backend/oamp/dynamics/optimization.py)) — multi-shooting on
  RK4 propagation, L-BFGS Hessian, accepts Lambert warm-starts
- TLE ingest with SGP4 ([`oamp.tle`](backend/oamp/tle.py)) — parse user-supplied two-line
  elements or fetch by NORAD ID from Celestrak
- Streaming WebSocket propagation (chunked JSON frames)
- WebGPU sandbox additions: Hohmann LEO→GEO scenario, Lambert intercept scenario,
  toggle-able **maneuver editor** sidebar (edit initial state, add/remove Δv kicks,
  hit *Propagate*), **time scrubber** in the footer that moves a spacecraft marker
  along the active trajectory

**Project conventions** (locked in to avoid bikeshedding later):
- **License**: Apache-2.0 (patent grant matters more than usual for aerospace)
- **Time scale**: UTC ISO-8601 on the wire, TDB seconds since J2000 (== SPICE ET) internally — see [`oamp.timescales`](backend/oamp/timescales.py)
- **Accelerator**: JAX (Numba dropped); enable via `pixi run -e gpu …`
- **Heavy compute**: Ray (Celery dropped) — see `pixi run -e full …`
- **SPICE kernels**: pinned by SHA256 in [`backend/oamp/spice/manifest.toml`](backend/oamp/spice/manifest.toml); fetcher prints hashes for first-run lock-in

## Pixi environments

- `default` — CPU-only backend (FastAPI, SciPy, SPICE, CasADi)
- `gpu`     — adds JAX (swap `jaxlib` for a CUDA build on Linux+NVIDIA)
- `full`    — adds Ray for batch GR workers

## Endpoints

- `GET  /health`              — version + status
- `POST /propagate`           — orbit propagation with selectable perturbations
  (zonals up to J6, third-body, drag, SRP), maneuvers, and integrator choice
  (`dop853` / `verlet` / `yoshida4`); `drag_model` may be `exponential` or `msis`
- `POST /launch`              — gravity-turn ascent + apoapsis circularization
- `GET  /launch/default-config`
- `POST /optimize/hohmann`    — closed-form coplanar two-impulse transfer
- `POST /optimize/lambert`    — universal-variable Lambert solver
- `POST /optimize/multi-burn` — CasADi/IPOPT fuel-optimal multi-burn (N ≥ 2)
- `POST /tle/parse`           — parse user-supplied TLE, optionally propagate to `at_utc`
- `GET  /tle/{norad}`         — fetch latest TLE from Celestrak, optional `?at_utc=...`
- `POST /spice/state`         — `(target, utc, observer, frame)` → r, v in SI
- `GET  /spice/status`        — which kernels are loaded
- `WS   /ws`                  — streaming propagator: `{"cmd":"propagate","request":...,"chunk_size":N}`

## License

Apache-2.0. See [LICENSE](LICENSE).
