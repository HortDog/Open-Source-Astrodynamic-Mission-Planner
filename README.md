# Open Source Astrodynamic Mission Planning (OAMP)

Interactive platform for space mission design with hybrid Newtonian/PN/GR dynamics,
SPICE-fidelity ephemerides, and a WebGPU sandbox frontend.

See the [Project Brief](Open_Source_Astrodynamics_Mission_Planning.pdf) and
[architecture diagram](docs/architecture.d2).

## Layout

- `backend/` — Python package `oamp` (FastAPI + dynamics engines)
- `web/` — Vite + TypeScript frontend (WebGPU)
- `docs/` — D2 architecture diagrams
- `pixi.toml` — Python environment and tasks
- `pyproject.toml` — Python package metadata

## Quick start

Backend (Python via pixi):

```sh
pixi install
pixi run test
pixi run dev          # http://localhost:8000
pixi run arch         # render docs/architecture.svg
```

Frontend (Vite):

```sh
cd web
npm install
npm run dev           # http://localhost:5173
```

## Pixi environments

- `default` — CPU-only backend (FastAPI, Numba, CasADi, SPICE)
- `gpu`     — adds JAX (swap `jaxlib` for a CUDA build on Linux+NVIDIA)
- `full`    — adds Ray for batch GR workers

```sh
pixi run -e gpu test
pixi run -e full dev
```

## License

Apache-2.0 (preferred over MIT for the explicit patent grant; see notes in
[docs/architecture.d2](docs/architecture.d2)).
