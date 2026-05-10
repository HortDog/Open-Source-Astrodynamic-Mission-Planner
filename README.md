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

**Phase 1 (this branch):**
- Newtonian propagator with optional J2 oblateness, DOP853 integrator
- Open-loop launch simulator (atmosphere drag, pitch program, apoapsis circularization)
- SPICE wrapper with reproducibility-locked kernel manifest
- WebGPU 3D viewer: orbit camera, central-body wireframe, axes, phase-colored trajectories
- Three demo scenarios in the UI: LEO orbit · Launch demo · LEO + J2 drift

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

- `GET  /health`           — version + status
- `POST /propagate`        — orbit propagation (Newton ± J2)
- `POST /launch`           — gravity-turn ascent + apoapsis circularization
- `GET  /launch/default-config`
- `POST /spice/state`      — `(target, utc, observer, frame)` → r, v in SI
- `GET  /spice/status`     — which kernels are loaded
- `WS   /ws`               — real-time channel (echo only for now)

## License

Apache-2.0. See [LICENSE](LICENSE).
