from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from oamp import __version__
from oamp.bodies import EARTH, MOON, SUN
from oamp.dynamics.integrators import propagate_symplectic
from oamp.dynamics.launch import LaunchConfig, default_falcon9_like, simulate_launch
from oamp.dynamics.newtonian import FiniteBurn, Maneuver, TwoBodyState, propagate_orbit
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


class SpiceStateRequest(BaseModel):
    target: str
    utc: str
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
    body = type(EARTH)(
        name="custom",
        mu=req.mu,
        radius=req.body_radius,
        j2=EARTH.j2,
        jn=EARTH.jn,
        omega=EARTH.omega,
    )
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
    body = type(EARTH)(
        name="custom",
        mu=req.mu,
        radius=req.body_radius,
        j2=EARTH.j2,
        jn=EARTH.jn,
        omega=EARTH.omega,
    )
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
