from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from oamp import __version__
from oamp.bodies import EARTH, MOON, SUN
from oamp.dynamics.cr3bp import (
    EM_LENGTH_M,
    EM_MEAN_MOTION_RAD_S,
    EM_MU,
    SE_MU,
    compute_manifold,
    find_planar_lyapunov,
    jacobi_constant,
    lagrange_points,
    propagate_cr3bp,
    wsb_capture_grid,
)
from oamp.dynamics.integrators import propagate_symplectic, propagate_symplectic_chunked
from oamp.dynamics.launch import LaunchConfig, default_falcon9_like, simulate_launch
from oamp.dynamics.newtonian import (
    FiniteBurn,
    Maneuver,
    TwoBodyState,
    propagate_orbit,
    propagate_orbit_chunked,
)
from oamp.dynamics.optimization import optimize_multi_burn
from oamp.dynamics.perturbations import (
    Vehicle,
    atmospheric_drag,
    compose,
    solar_radiation_pressure,
    third_body,
    zonal_harmonics,
)
from oamp.dynamics.transfers import hohmann_transfer, lambert_universal
from oamp.tle import fetch_celestrak, jd_from_iso_utc, propagate_tle, tle_state

# --------------------------------------------------------------------------- #
#  Request / response models
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    status: str
    version: str


class VehicleModel(BaseModel):
    mass_kg: float = Field(gt=0)
    drag_area_m2: float = 1.0
    drag_cd: float = 2.2
    srp_area_m2: float = 1.0
    srp_cr: float = 1.5


class ManeuverModel(BaseModel):
    t_offset_s: float = Field(ge=0)
    dv_ric: tuple[float, float, float]


class FiniteBurnModel(BaseModel):
    t_start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    thrust_n: float = Field(gt=0)
    isp_s: float = Field(gt=0)
    direction_ric: tuple[float, float, float] = (0.0, 1.0, 0.0)


class PropagateRequest(BaseModel):
    state: TwoBodyState
    duration_s: float
    steps: Annotated[int, Field(ge=2, le=20_000)] = 200
    # Central-body selection. If `body_name` is supplied, μ / radius / J2 / Jn / ω
    # are loaded from the matching Body constant and `mu` / `body_radius` are
    # ignored (kept for back-compat with old clients).
    body_name: Literal["EARTH", "MOON", "SUN"] | None = None
    mu: float = EARTH.mu
    body_radius: float = EARTH.radius
    # Perturbation toggles
    j2_enabled: bool = False  # back-compat shortcut
    jn_max: Annotated[int, Field(ge=2, le=6)] | None = None
    drag: bool = False
    srp: bool = False
    third_body: list[str] = []  # e.g. ["MOON", "SUN"]
    vehicle: VehicleModel | None = None
    t0_tdb: float = 0.0
    maneuvers: list[ManeuverModel] = []
    finite_burns: list[FiniteBurnModel] = []
    initial_mass_kg: float | None = Field(default=None, gt=0)
    integrator: Literal["dop853", "verlet", "yoshida4"] = "dop853"
    # Drag model: 'exponential' (default, no extra deps) or 'msis' (pymsis).
    drag_model: Literal["exponential", "msis"] = "exponential"


class HohmannRequest(BaseModel):
    r1_m: float = Field(gt=0)
    r2_m: float = Field(gt=0)
    mu: float = EARTH.mu


class LambertRequest(BaseModel):
    r1_m: tuple[float, float, float]
    r2_m: tuple[float, float, float]
    tof_s: float = Field(gt=0)
    mu: float = EARTH.mu
    prograde: bool = True


class MultiBurnRequest(BaseModel):
    x0_r: tuple[float, float, float]
    x0_v: tuple[float, float, float]
    xf_r: tuple[float, float, float]
    xf_v: tuple[float, float, float]
    maneuver_epochs_s: list[float]
    t_final_s: float = Field(gt=0)
    mu: float = EARTH.mu
    initial_dv_guess: list[tuple[float, float, float]] | None = None


class Cr3bpPropagateRequest(BaseModel):
    """Propagate a state in the non-dimensional CR3BP."""

    state: tuple[float, float, float, float, float, float]
    t_span: tuple[float, float]
    mu: float = Field(gt=0, lt=0.5, default=EM_MU)
    steps: Annotated[int, Field(ge=2, le=20_000)] = 400


class TransformStatesRequest(BaseModel):
    """Batch frame transform.  `direction` is either 'to_synodic' or
    'from_synodic'; `frame` is the target/source rotating frame.

    `t_offsets_s`, if provided, must match the length of `states`: each state
    is transformed using ``t_tdb + t_offsets_s[i]``.  Without it, all states
    use the same `t_tdb` — a "frozen-frame" view.  The rotating-frame view
    that makes the Moon appear stationary needs per-state offsets."""

    direction: Literal["to_synodic", "from_synodic"] = "to_synodic"
    frame: Literal["EM_SYNODIC"] = "EM_SYNODIC"
    t_tdb: float
    t_offsets_s: list[float] | None = None
    states: list[tuple[float, float, float, float, float, float]]


class Cr3bpPeriodicOrbitRequest(BaseModel):
    """Differential-correction request for a CR3BP periodic orbit."""

    family: Literal["lyapunov"] = "lyapunov"
    L_point: Literal[1, 2] = 1
    Ax: float = Field(gt=0, lt=0.2, default=0.01)
    mu: float = Field(gt=0, lt=0.5, default=EM_MU)


class Cr3bpManifoldRequest(BaseModel):
    """Manifold tube computation for a CR3BP periodic orbit."""

    orbit_state: tuple[float, float, float, float, float, float]
    period: float = Field(gt=0)
    mu: float = Field(gt=0, lt=0.5, default=EM_MU)
    direction: Literal["stable", "unstable"] = "unstable"
    branch: Literal["+", "-"] = "+"
    n_samples: Annotated[int, Field(ge=4, le=200)] = 40
    duration: float = Field(gt=0, le=20, default=8.0)
    perturbation: float = Field(gt=0, default=1e-6)
    steps: Annotated[int, Field(ge=20, le=2000)] = 200


class WsbGridRequest(BaseModel):
    """Weak-Stability-Boundary diagnostic grid request."""

    altitudes_m: list[float]
    angles_rad: list[float]
    mu: float = Field(gt=0, lt=0.5, default=EM_MU)
    duration: float = Field(gt=0, le=20, default=6.0)
    escape_radius: float = Field(gt=0, default=2.0)


class SpiceStateRequest(BaseModel):
    target: str
    utc: str
    observer: str = "EARTH"
    frame: str = "J2000"


class SpiceEphemerisRequest(BaseModel):
    """Batched ephemeris query: list of TDB-second offsets from `t0_utc`."""

    target: str
    t0_utc: str
    t_offsets_s: list[float]
    observer: str = "EARTH"
    frame: str = "J2000"


class TleRequest(BaseModel):
    """Submit a TLE directly (e.g. for an internal catalogue). For Celestrak
    fetches use ``GET /tle?norad=...`` instead."""

    line1: str
    line2: str
    name: str = ""
    at_utc: str | None = None  # if supplied, propagate to this epoch


# --------------------------------------------------------------------------- #
#  App lifecycle
# --------------------------------------------------------------------------- #


_BODY_MU = {"EARTH": EARTH.mu, "MOON": MOON.mu, "SUN": SUN.mu}
_BODIES = {"EARTH": EARTH, "MOON": MOON, "SUN": SUN}


def _resolve_body(req: PropagateRequest):
    """Return the Body to use for a propagate request.

    When `body_name` is set, return the canonical Body constant (so J_n, ω,
    and J2 match the chosen central body).  Otherwise build a custom Body
    around the legacy `mu`/`body_radius` fields with Earth's J_n profile
    (back-compat path used by the existing demos)."""
    if req.body_name is not None:
        return _BODIES[req.body_name]
    return type(EARTH)(
        name="custom",
        mu=req.mu,
        radius=req.body_radius,
        j2=EARTH.j2,
        jn=EARTH.jn,
        omega=EARTH.omega,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from oamp import spice

        loaded = spice.furnsh_dir()
        app.state.spice_kernels = [str(p.name) for p in loaded]
    except Exception as e:
        app.state.spice_kernels = []
        app.state.spice_error = str(e)
    yield


app = FastAPI(title="OAMP", version=__version__, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  Status endpoints
# --------------------------------------------------------------------------- #


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.get("/spice/status")
async def spice_status() -> dict:
    return {
        "loaded_kernels": getattr(app.state, "spice_kernels", []),
        "error": getattr(app.state, "spice_error", None),
    }


# --------------------------------------------------------------------------- #
#  Propagation
# --------------------------------------------------------------------------- #


def _build_perturbation(req: PropagateRequest, body) -> tuple[object | None, list[str]]:
    """Compose the perturbation list from the request flags. Returns the
    callable and a list of strings naming what was active (for the response)."""
    perturbations: list[object] = []
    active: list[str] = []

    n_max = req.jn_max or (2 if req.j2_enabled else 0)
    if n_max >= 2:
        perturbations.append(zonal_harmonics(body, n_max=n_max))
        active.append(f"zonal_J2_to_J{n_max}")

    for name in req.third_body:
        upper = name.upper()
        if upper not in _BODY_MU:
            raise HTTPException(status_code=400, detail=f"unknown third body: {name}")
        perturbations.append(third_body(upper, _BODY_MU[upper]))
        active.append(f"third_body_{upper}")

    if req.drag or req.srp:
        if req.vehicle is None:
            raise HTTPException(
                status_code=400,
                detail="drag/srp require a `vehicle` block (mass, area, Cd/Cr)",
            )
        veh = Vehicle(**req.vehicle.model_dump())
        if req.drag:
            if req.drag_model == "msis":
                try:
                    from oamp.dynamics.atmosphere import msis_density_fn

                    perturbations.append(
                        atmospheric_drag(veh, body=body, density_fn_full=msis_density_fn())
                    )
                    active.append("drag_msis")
                except ImportError as e:
                    raise HTTPException(status_code=503, detail=str(e)) from e
            else:
                perturbations.append(atmospheric_drag(veh, body=body))
                active.append("drag_exponential")
        if req.srp:
            perturbations.append(solar_radiation_pressure(veh, body_radius_m=body.radius))
            active.append("srp_cannonball")

    if not perturbations:
        return None, []
    return compose(*perturbations), active


@app.post("/propagate")
async def propagate(req: PropagateRequest) -> dict:
    body = _resolve_body(req)
    perturbation, active = _build_perturbation(req, body)
    maneuvers = [Maneuver(**m.model_dump()) for m in req.maneuvers]
    fburns = [FiniteBurn(**b.model_dump()) for b in req.finite_burns]
    if fburns:
        active.append(f"finite_burns_{len(fburns)}")
    try:
        if req.integrator in ("verlet", "yoshida4"):
            if fburns:
                raise HTTPException(
                    status_code=400,
                    detail="finite burns require the dop853 integrator",
                )
            times, states = propagate_symplectic(
                req.state,
                req.duration_s,
                req.steps,
                body=body,
                perturbation=perturbation,
                j2_enabled=req.j2_enabled and perturbation is None,
                maneuvers=maneuvers,
                t0_tdb=req.t0_tdb,
                method=req.integrator,
            )
            active.append(f"integrator_{req.integrator}")
            return {"t": times.tolist(), "states": states.tolist(), "perturbations": active}
        times, states = propagate_orbit(
            req.state,
            req.duration_s,
            req.steps,
            body=body,
            j2_enabled=req.j2_enabled and perturbation is None,
            perturbation=perturbation,
            maneuvers=maneuvers,
            finite_burns=fburns or None,
            initial_mass_kg=req.initial_mass_kg,
            t0_tdb=req.t0_tdb,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "t": times.tolist(),
        "states": states.tolist(),
        "perturbations": active,
    }


@app.post("/propagate/stream")
async def propagate_stream(req: PropagateRequest) -> StreamingResponse:
    """Like /propagate but streams partial trajectory chunks as NDJSON.

    Each line is a JSON object:
      - partial chunk: {"t":[...], "states":[[...]], "received_steps":N, "total_steps":M}
      - final line:    {"done":true, "perturbations":[...]}
    """
    body = _resolve_body(req)
    try:
        perturbation, active = _build_perturbation(req, body)
    except HTTPException as exc:
        raise exc
    maneuvers = [Maneuver(**m.model_dump()) for m in req.maneuvers]
    fburns = [FiniteBurn(**b.model_dump()) for b in req.finite_burns]
    if fburns:
        active.append(f"finite_burns_{len(fburns)}")

    # Target ~20 chunks regardless of step count; clamp to sensible range.
    chunk_size = max(50, min(500, req.steps // 20))

    async def _gen():
        received = 0
        try:
            if req.integrator in ("verlet", "yoshida4"):
                if fburns:
                    yield json.dumps({"error": "finite burns require the dop853 integrator"}) + "\n"
                    return
                gen = propagate_symplectic_chunked(
                    req.state, req.duration_s, req.steps,
                    body=body, perturbation=perturbation,
                    j2_enabled=req.j2_enabled and perturbation is None,
                    maneuvers=maneuvers, t0_tdb=req.t0_tdb,
                    method=req.integrator, chunk_size=chunk_size,
                )
                active.append(f"integrator_{req.integrator}")
            else:
                gen = propagate_orbit_chunked(
                    req.state, req.duration_s, req.steps,
                    body=body, perturbation=perturbation,
                    j2_enabled=req.j2_enabled and perturbation is None,
                    maneuvers=maneuvers,
                    finite_burns=fburns or None,
                    initial_mass_kg=req.initial_mass_kg,
                    t0_tdb=req.t0_tdb,
                    chunk_size=chunk_size,
                )

            for t_chunk, s_chunk in gen:
                received += len(t_chunk)
                msg = {
                    "t": t_chunk.tolist(),
                    "states": s_chunk.tolist(),
                    "received_steps": received,
                    "total_steps": req.steps,
                }
                yield json.dumps(msg) + "\n"
                await asyncio.sleep(0)  # yield to event loop between chunks

            yield json.dumps({"done": True, "perturbations": active}) + "\n"
        except Exception as exc:
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(_gen(), media_type="application/x-ndjson")


@app.post("/launch")
async def launch(config: LaunchConfig | None = None) -> dict:
    cfg = config or default_falcon9_like()
    return simulate_launch(cfg).model_dump()


@app.get("/launch/default-config")
async def launch_default_config() -> dict:
    return default_falcon9_like().model_dump()


# --------------------------------------------------------------------------- #
#  Transfer solvers
# --------------------------------------------------------------------------- #


@app.post("/optimize/hohmann")
async def optimize_hohmann(req: HohmannRequest) -> dict:
    res = hohmann_transfer(req.r1_m, req.r2_m, mu=req.mu)
    return {
        "dv1_m_s": res.dv1_m_s,
        "dv2_m_s": res.dv2_m_s,
        "dv_total_m_s": res.dv_total_m_s,
        "transfer_time_s": res.transfer_time_s,
        "semi_major_axis_m": res.semi_major_axis_m,
    }


@app.post("/optimize/lambert")
async def optimize_lambert(req: LambertRequest) -> dict:
    try:
        res = lambert_universal(
            req.r1_m,
            req.r2_m,
            req.tof_s,
            mu=req.mu,
            prograde=req.prograde,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "v1_m_s": res.v1_m_s,
        "v2_m_s": res.v2_m_s,
        "iterations": res.iterations,
        "converged": res.converged,
        "transfer_time_s": res.transfer_time_s,
    }


@app.post("/optimize/multi-burn")
async def optimize_multi(req: MultiBurnRequest) -> dict:
    try:
        res = optimize_multi_burn(
            x0_r=req.x0_r,
            x0_v=req.x0_v,
            xf_r=req.xf_r,
            xf_v=req.xf_v,
            maneuver_epochs_s=req.maneuver_epochs_s,
            t_final_s=req.t_final_s,
            mu=req.mu,
            initial_dv_guess=req.initial_dv_guess,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"solver error: {e}") from e
    return {
        "dv_inertial_m_s": res.dv_inertial_m_s,
        "total_dv_m_s": res.total_dv_m_s,
        "converged": res.converged,
        "iterations": res.iterations,
        "final_state_m": res.final_state_m,
        "final_velocity_m_s": res.final_velocity_m_s,
    }


# --------------------------------------------------------------------------- #
#  CR3BP / rotating frames (Phase 4)
# --------------------------------------------------------------------------- #

_CR3BP_PRESETS = {"EARTH_MOON": EM_MU, "SUN_EARTH": SE_MU}


@app.get("/cr3bp/lagrange")
async def cr3bp_lagrange(
    mu: float | None = None,
    system: Literal["EARTH_MOON", "SUN_EARTH"] | None = None,
) -> dict:
    """Return the five Lagrange points (non-dim) for the given mass ratio.

    Either `mu` or `system` must be supplied (`system` selects a built-in
    preset that resolves to a μ from the canonical Body constants)."""
    if mu is None and system is None:
        raise HTTPException(status_code=400, detail="provide mu= or system=")
    resolved_mu = mu if mu is not None else _CR3BP_PRESETS[system]  # type: ignore[index]
    try:
        L = lagrange_points(resolved_mu)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "mu": resolved_mu,
        "L1": L.L1, "L2": L.L2, "L3": L.L3, "L4": L.L4, "L5": L.L5,
    }


@app.post("/cr3bp/propagate")
async def cr3bp_propagate(req: Cr3bpPropagateRequest) -> dict:
    """Propagate a non-dimensional CR3BP state and return the trajectory plus
    the Jacobi-constant trace (a free integration-error diagnostic)."""
    import numpy as np

    state0 = np.asarray(req.state, dtype=float)
    try:
        t, states = propagate_cr3bp(state0, req.t_span, req.mu, steps=req.steps)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    C = [jacobi_constant(s, req.mu) for s in states]
    return {
        "t": t.tolist(),
        "states": states.tolist(),
        "jacobi": C,
        "mu": req.mu,
    }


@app.post("/transform/states")
async def transform_states(req: TransformStatesRequest) -> dict:
    """Convert a batch of states between J2000 (Earth-centric) and the
    Earth–Moon synodic frame at the given TDB epoch.  Requires SPICE."""
    try:
        from oamp.frames import em_synodic_to_inertial, inertial_to_em_synodic
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"frames unavailable: {e}") from e

    import numpy as np

    if req.t_offsets_s is not None and len(req.t_offsets_s) != len(req.states):
        raise HTTPException(
            status_code=400,
            detail=f"t_offsets_s length {len(req.t_offsets_s)} != states {len(req.states)}",
        )
    out: list[list[float]] = []
    try:
        for i, s in enumerate(req.states):
            r = np.asarray(s[:3], dtype=float)
            v = np.asarray(s[3:], dtype=float)
            t_i = req.t_tdb + (req.t_offsets_s[i] if req.t_offsets_s else 0.0)
            if req.direction == "to_synodic":
                rr, vv = inertial_to_em_synodic(r, v, t_i)
            else:
                rr, vv = em_synodic_to_inertial(r, v, t_i)
            out.append([*rr.tolist(), *vv.tolist()])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "frame": req.frame,
        "direction": req.direction,
        "t_tdb": req.t_tdb,
        "states": out,
        "length_scale_m": EM_LENGTH_M,
        "mean_motion_rad_s": EM_MEAN_MOTION_RAD_S,
    }


@app.post("/cr3bp/periodic-orbit")
async def cr3bp_periodic_orbit(req: Cr3bpPeriodicOrbitRequest) -> dict:
    """Run differential correction for a planar Lyapunov orbit around L1 or L2.

    Returns the converged IC, full orbit period, Jacobi constant, and DC
    convergence info; the caller can feed the IC into `/cr3bp/propagate` to
    visualise the closed orbit."""
    if req.family != "lyapunov":
        raise HTTPException(status_code=400, detail=f"unsupported family: {req.family}")
    try:
        orbit = find_planar_lyapunov(L_point=req.L_point, Ax=req.Ax, mu=req.mu)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "state0": list(orbit.state0),
        "period": orbit.period,
        "jacobi": orbit.jacobi,
        "family": orbit.family,
        "dc_iterations": orbit.dc_iterations,
        "dc_residual": orbit.dc_residual,
        "mu": req.mu,
    }


@app.post("/cr3bp/manifold")
async def cr3bp_manifold(req: Cr3bpManifoldRequest) -> dict:
    """Compute one branch of an invariant manifold for a periodic orbit.

    Returns ``n_samples`` trajectory tubes, each shape ``(steps, 6)``.  Some
    may be empty (None entries) if individual integrations failed near the
    primaries."""
    import numpy as np

    try:
        tubes = compute_manifold(
            np.asarray(req.orbit_state, dtype=float),
            req.period,
            req.mu,
            direction=req.direction,
            branch=req.branch,
            n_samples=req.n_samples,
            duration=req.duration,
            perturbation=req.perturbation,
            steps=req.steps,
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "mu": req.mu,
        "direction": req.direction,
        "branch": req.branch,
        "trajectories": [t.tolist() for t in tubes],
    }


@app.post("/cr3bp/wsb")
async def cr3bp_wsb(req: WsbGridRequest) -> dict:
    """Crude Weak-Stability-Boundary capture/escape classification grid.

    Returns a (len(altitudes_m), len(angles_rad)) integer grid where
    1 = captured-from-past, 0 = escaped, −1 = integrator failure."""
    import numpy as np

    if len(req.altitudes_m) * len(req.angles_rad) > 5_000:
        raise HTTPException(status_code=400, detail="grid too large (max 5000 cells)")
    grid = wsb_capture_grid(
        np.asarray(req.altitudes_m),
        np.asarray(req.angles_rad),
        mu=req.mu,
        duration=req.duration,
        escape_radius=req.escape_radius,
    )
    return {
        "mu": req.mu,
        "altitudes_m": req.altitudes_m,
        "angles_rad": req.angles_rad,
        "grid": grid.tolist(),
    }


# --------------------------------------------------------------------------- #
#  TLE ingest
# --------------------------------------------------------------------------- #


def _tle_state_response(line1: str, line2: str, name: str, at_utc: str | None) -> dict:
    from oamp.tle import parse_tle  # local import

    sat = parse_tle(line1, line2, name)
    if at_utc is None:
        ts = tle_state(line1, line2, name=name)
        jd_used = ts.epoch_jd
        r = ts.r_m
        v = ts.v_m_s
    else:
        jd, fr = jd_from_iso_utc(at_utc)
        r, v, jd_used = propagate_tle(sat, jd, fr)
    speed = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    radius = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5
    return {
        "name": name,
        "norad_id": int(sat.satnum),
        "epoch_jd": jd_used,
        "state": {"r": list(r), "v": list(v)},
        "altitude_km": (radius - EARTH.radius) / 1000.0,
        "speed_m_s": speed,
    }


@app.post("/tle/parse")
async def tle_parse(req: TleRequest) -> dict:
    """Parse a user-supplied TLE; optionally propagate to ``at_utc``."""
    try:
        return _tle_state_response(req.line1, req.line2, req.name, req.at_utc)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/tle/{norad}")
async def tle_fetch(norad: int, at_utc: str | None = None) -> dict:
    """Fetch the latest TLE for a NORAD catalogue number from Celestrak."""
    try:
        name, line1, line2 = fetch_celestrak(norad)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Celestrak fetch failed: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        return _tle_state_response(line1, line2, name, at_utc)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# --------------------------------------------------------------------------- #
#  SPICE state
# --------------------------------------------------------------------------- #


@app.post("/spice/state")
async def spice_state(req: SpiceStateRequest) -> dict:
    try:
        from oamp import spice
        from oamp.timescales import utc_iso_to_et
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"SPICE unavailable: {e}") from e
    try:
        et = utc_iso_to_et(req.utc)
        r, v = spice.body_state(req.target, et, req.observer, req.frame)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "et": et,
        "r": r.tolist(),
        "v": v.tolist(),
        "frame": req.frame,
        "observer": req.observer,
    }


@app.post("/spice/ephemeris")
async def spice_ephemeris(req: SpiceEphemerisRequest) -> dict:
    """Query `target` ephemeris at `t0_utc + dt` for each dt in `t_offsets_s`.

    Returns one (r, v) pair per offset.  Limit: 5_000 samples per request to
    keep response sizes manageable."""
    if len(req.t_offsets_s) > 5_000:
        raise HTTPException(status_code=400, detail="too many samples (max 5000)")
    try:
        from oamp import spice
        from oamp.timescales import utc_iso_to_et
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"SPICE unavailable: {e}") from e
    try:
        et0 = utc_iso_to_et(req.t0_utc)
        rs: list[list[float]] = []
        vs: list[list[float]] = []
        for dt in req.t_offsets_s:
            r, v = spice.body_state(req.target, et0 + float(dt), req.observer, req.frame)
            rs.append(r.tolist())
            vs.append(v.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"et0": et0, "r": rs, "v": vs, "frame": req.frame, "observer": req.observer}


# --------------------------------------------------------------------------- #
#  WebSocket — streaming propagation
# --------------------------------------------------------------------------- #


WSCommand = Literal["propagate", "echo"]


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Streaming WebSocket. Client sends a JSON command of the form:

        {"cmd": "propagate", "request": <PropagateRequest>, "chunk_size": 50}

    The server runs the propagation in arc-chunks and emits each chunk as a
    JSON frame {"chunk_index": i, "t": [...], "states": [...]}, then a final
    {"done": true, "elapsed_ms": ...} sentinel. Other commands echo.
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid json"})
                continue

            cmd = msg.get("cmd", "echo")
            if cmd == "propagate":
                try:
                    req = PropagateRequest(**msg["request"])
                except Exception as e:
                    await websocket.send_json({"error": f"bad request: {e}"})
                    continue
                await _stream_propagation(websocket, req, msg.get("chunk_size", 50))
            else:
                await websocket.send_json({"echo": msg})
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        return


async def _stream_propagation(
    ws: WebSocket,
    req: PropagateRequest,
    chunk_size: int,
) -> None:
    import time

    t_start = time.perf_counter()
    body = _resolve_body(req)
    perturbation, active = _build_perturbation(req, body)
    maneuvers = [Maneuver(**m.model_dump()) for m in req.maneuvers]

    # Single-shot propagate, then stream the result in chunks. This keeps the
    # heavy compute synchronous (SciPy ODE) but yields back to the event loop
    # between chunks so the WebSocket stays responsive to cancellation.
    try:
        times, states = propagate_orbit(
            req.state,
            req.duration_s,
            req.steps,
            body=body,
            j2_enabled=req.j2_enabled and perturbation is None,
            perturbation=perturbation,
            maneuvers=maneuvers,
            t0_tdb=req.t0_tdb,
        )
    except Exception as e:
        await ws.send_json({"error": str(e)})
        return

    await ws.send_json({"meta": {"total_steps": len(times), "perturbations": active}})

    n = len(times)
    for start in range(0, n, max(1, chunk_size)):
        end = min(start + chunk_size, n)
        await ws.send_json(
            {
                "chunk_index": start // chunk_size,
                "t": times[start:end].tolist(),
                "states": states[start:end].tolist(),
            }
        )
        await asyncio.sleep(0)

    await ws.send_json({"done": True, "elapsed_ms": (time.perf_counter() - t_start) * 1000})
