# Open Source Astrodynamic Mission Planning (OAMP)

Interactive platform for space-mission design with high-fidelity Newtonian
dynamics, SPICE ephemerides, low-energy / CR3BP transfer design, and a
phosphor-styled WebGPU sandbox frontend. AGPLv3.

## Layout

- `backend/` — Python package `oamp` (FastAPI + dynamics engines)
  - `oamp.api`        — FastAPI app, REST + WebSocket + NDJSON streaming
  - `oamp.bodies`     — central-body constants (μ, R, J₂…J₆, ω)
  - `oamp.dynamics`   — `newtonian`, `integrators` (Verlet/Yoshida), `perturbations`, `cr3bp`, `frames`, `transfers`, `optimization`, `launch`, `atmosphere` (MSIS)
  - `oamp.spice`      — kernel manifest + SHA256-verified downloader + SI-unit query wrapper
  - `oamp.tle`        — SGP4 + Celestrak fetcher
  - `oamp.timescales` — UTC ↔ TDB ET conversions
- `web/` — Vite + TypeScript frontend (WebGPU)
  - `src/render/scene.ts`           — WebGPU thick-line renderer
  - `src/render/earth-coastline.ts` — graticule + Natural Earth coastline loader
  - `src/kepler.ts`                 — Cartesian ↔ classical orbital elements
- `data/kernels/` — SPICE kernels (DE440s + LSK + PCK). **Tracked in the repo so a fresh clone runs without network fetches.**
- `web/public/data/` — Natural Earth 110m coastline GeoJSON. **Tracked in the repo.**
- `docs/` — LaTeX roadmap
- `pixi.toml` — Python environment and tasks

## Quick start

```sh
pixi install
pixi run test           # backend tests (Newtonian, CR3BP, transfers, time scales)
pixi run dev            # http://localhost:8000  (FastAPI + /docs)
```

```sh
cd web && npm install
npm run dev             # http://localhost:5173
```

A fresh clone has every kernel and the coastline already in-tree — no
network fetches needed on first run. `pixi run kernels` is only needed when
bumping kernel versions (and even then, only on the dev box; CI verifies
sha256 against the manifest).

## What's shipped

**Phase 1 — Foundations.** Newtonian propagator with J₂, DOP853 integrator,
open-loop launch simulator, SPICE wrapper with SHA256-pinned kernel manifest,
WebGPU 3D viewer with orbit camera and phase-coloured trajectories.

**Phase 2 — Precision Newtonian.** Pluggable perturbation engine
(zonal J₂–J₆, third-body Moon/Sun via SPICE, cannonball SRP with cylindrical
shadow, exponential + MSIS drag); symplectic integrators (Verlet, Yoshida-4);
impulsive Δv in RIC + finite burns; Lambert universal-variable solver;
closed-form Hohmann; CasADi/IPOPT multi-burn NLP; SGP4 TLE ingest; WebSocket
streaming; maneuver editor + time scrubber UI.

**Phase 3 — Sandbox integration & lunar system.** Solver panel, central-body
selector (Earth/Moon with canonical μ/R/J/ω), SPICE-driven Moon rendering as
a secondary body, end-to-end cislunar TLI demo (Lambert + J₂ + 3rd-body Moon
+ finite-burn-ready vehicle), perturbation editor, apse markers, altitude
chart, animated time scrubber.

**Phase 4 — CR3BP & low-energy transfers.** Non-dimensional circular
restricted three-body propagator (Earth–Moon and Sun–Earth presets), L₁–L₅
computation, Jacobi-constant trace, planar Lyapunov periodic orbits via
differential correction, monodromy / invariant manifold tubes, weak-stability
boundary (WSB) capture diagnostic, EM-synodic frame transforms.

**Mission planner polish.** Full custom-mission editor surfacing every
backend feature: integrator selector, t₀ epoch picker, SPICE kernel-status
badge, 9-orbit preset library (ISS / LEO / SSO / GTO / GEO / Molniya / lunar
/ EML1+2 halo seeds), live Δv budget, cancel-streaming control, save/load
mission JSON + autosave to `localStorage`, CSV trajectory export, orbital
elements editor (Cartesian ↔ Keplerian), SVG event timeline, inline
validation pills, collapsible sections, recenter camera, footer chart
selector (|r| / |v| / specific energy), in-editor CR3BP / Lyapunov–manifold
/ WSB / launch panels, multi-burn NLP inertial→RIC auto-conversion,
play/pause animation with ×1…×10M speed range, past-as-solid / future-as-
dashed trajectory rendering, Natural Earth coastline overlay on the Earth
wireframe.

## What's coming next

**Phase 5 — Real star field & astro-navigation.** Render a real celestial
sphere from a star catalogue (Hipparcos / Yale Bright Star) so the
background isn't black; then build navigation algorithms that *use* it the
way real spacecraft do.

- WebGPU point-sprite renderer for ≥10,000 stars by RA/Dec with apparent
  magnitude and B–V colour
- Constellation outlines (toggle-able)
- Sun, Moon, planet positions from SPICE rendered as bright movers
- Star tracker simulation: spacecraft body-frame attitude → which stars are
  in the FOV, with noise model and pixel positions
- Lost-in-space attitude solver (triangle / pyramid star-ID algorithms)
- Optical navigation: simulate landmark / planet-limb measurements and
  estimate spacecraft state via least-squares / EKF
- Sun-sensor + Earth-horizon-sensor + magnetometer models for cubesat ADCS
  studies
- "Navigate by stars" demo mode — close the loop with a star tracker driving
  an estimator that corrects an artificially noisy propagator

See [docs/roadmap_phase_plan.tex](docs/roadmap_phase_plan.tex) for the full
work breakdown.

## Future / stretch goals

These are deliberately ambitious and unscheduled. PRs welcome.

- **General Relativity.** Post-Newtonian Schwarzschild + Lense–Thirring +
  de Sitter corrections, then a full BCRS geodesic integrator with tetrad
  thrust and a Ray-backed metric pre-bake cache. The right answer for
  pulsar timing, geodesy, deep-space precision navigation.
- **Tesseral gravity** (C/S terms beyond the zonals) for accurate GEO
  station-keeping and Earth–Moon mascon work.
- **Ground tracks + visibility analysis** — pass prediction, station
  contact windows, eclipse maps.
- **Conjunction analysis** against a TLE catalogue.

## Let your creativity go wild

Open-ended directions where the only constraint is "it should be cool":

- **Tour mode** — auto-fly the camera through historical missions (Apollo
  11 TLI, Voyager Jupiter encounter, Hayabusa-2 touchdown, JWST L₂ insertion)
  with annotated overlays
- **Live-ISS mode** — pull the current ISS TLE every few hours and render
  its present position with a sun-illumination indicator
- **Constellation builder** — drop a Walker / Flower / star-of-stars
  constellation, propagate everything, render coverage maps
- **Generative trajectory designer** — natural-language → trajectory
  ("get me to Europa for under 5 km/s") with an LLM driving the solver chain
- **Asteroid-mining sandbox** — pull JPL Horizons NEAs, plan rendezvous,
  return-trajectory ΔV budgets, payload mass fractions
- **Solar-sail / electric-propulsion** — low-thrust spiraling, Edelbaum,
  shape-based methods; tightly coupled to a power model that follows SRP
  shadow and orientation
- **WebXR mode** — view the scene in a headset; reach into the orbit and
  drag the spacecraft to retime a maneuver
- **AR through your phone** — point camera at the night sky, overlay
  predicted satellite paths from our propagator
- **Audio synthesis** — drive a synth from orbital frequencies, eccentricity
  modulation, eclipses-as-amplitude — Kepler's "music of the spheres" but
  literal
- **Procedural mission cinematography** — render a 60-second mp4 from any
  trajectory with camera moves, captions, and a music bed
- **Lyapunov-spectrum visualiser** — colour each piece of state space by
  its local Lyapunov exponent so chaos around L₃ becomes visible
- **Alternate-physics sandbox** — "what if G was 2×?", "what if the Moon
  weren't there?", "what if Mercury orbited with PN turned off?" — change
  the constants, re-propagate, watch the orbit drift
- **Mission narrative export** — pull a 2-paragraph trip summary from the
  propagator data: "departed LEO 2026-01-04 17:30Z, performed a 3.1 km/s
  TLI burn, encountered the lunar SOI 4.2 days later…"
- **In-browser optimisation playground** — drag a target marker around and
  watch the multi-burn NLP re-solve in real time
- **Deep-field renderer** — JWST-style starfield with diffraction spikes,
  rendered when the camera looks away from any solar-system body

If a wild idea is missing from this list, add it and open a PR.

## Project conventions

- **Licence**: AGPL-3.0. See [licence.txt](licence.txt). Network-service use
  triggers the source-disclosure obligation — important to be aware of
  before deploying a hosted instance.
- **Time scale**: UTC ISO-8601 on the wire, TDB seconds since J2000 (==
  SPICE ET) internally — see [`oamp.timescales`](backend/oamp/timescales.py)
- **Units**: SI throughout the engine (m, m/s, kg, s)
- **Accelerator**: JAX (`pixi run -e gpu …`); CPU default
- **Heavy compute**: Ray (`pixi run -e full …`)
- **SPICE kernels**: pinned by SHA256 in [`backend/oamp/spice/manifest.toml`](backend/oamp/spice/manifest.toml); see [`data/kernels/README.md`](data/kernels/README.md) for regenerate instructions

## Pixi environments

- `default` — CPU-only backend (FastAPI, SciPy, SPICE, CasADi, pymsis)
- `gpu`     — adds JAX (swap `jaxlib` for a CUDA build on Linux+NVIDIA)
- `full`    — adds Ray for batch / heavy-compute jobs

## Endpoints

- `GET  /health`                  — version + status
- `GET  /spice/status`            — which SPICE kernels are loaded
- `POST /propagate`               — full-fidelity orbit propagation (zonals J₂–J₆, third-body, drag, SRP, finite burns, integrator choice, drag-model choice)
- `POST /propagate/stream`        — same, NDJSON-streamed chunks for progressive UI rendering
- `POST /launch`                  — gravity-turn ascent + apoapsis circularization
- `GET  /launch/default-config`   — default Falcon-9-like vehicle
- `POST /optimize/hohmann`        — closed-form coplanar two-impulse transfer
- `POST /optimize/lambert`        — universal-variable Lambert solver
- `POST /optimize/multi-burn`     — CasADi / IPOPT fuel-optimal multi-burn NLP
- `GET  /cr3bp/lagrange`          — L₁–L₅ positions for a given mass ratio
- `POST /cr3bp/propagate`         — non-dim CR3BP integration with Jacobi-constant trace
- `POST /cr3bp/periodic-orbit`    — Lyapunov differential correction
- `POST /cr3bp/manifold`          — invariant manifold tube generation
- `POST /cr3bp/wsb`               — weak-stability boundary capture grid
- `POST /transform/states`        — frame transforms (J2000 ↔ EM_SYNODIC)
- `POST /tle/parse`               — parse user-supplied TLE; optionally propagate to `at_utc`
- `GET  /tle/{norad}`             — fetch latest TLE from Celestrak
- `POST /spice/state`             — `(target, utc, observer, frame)` → r, v in SI
- `POST /spice/ephemeris`         — batched ephemeris query
- `WS   /ws`                      — legacy streaming propagator (kept; new code uses `/propagate/stream`)

## License

AGPL-3.0. See [licence.txt](licence.txt).
