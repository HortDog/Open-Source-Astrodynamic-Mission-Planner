from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from oamp import __version__
from oamp.bodies import EARTH, MOON, SUN
from oamp.dynamics.launch import LaunchConfig, default_falcon9_like, simulate_launch
from oamp.dynamics.newtonian import Maneuver, TwoBodyState, propagate_orbit
from oamp.dynamics.perturbations import (
    Vehicle,
    atmospheric_drag,
    compose,
    solar_radiation_pressure,
    third_body,
    zonal_harmonics,
)
from oamp.dynamics.transfers import hohmann_transfer, lambert_universal

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


class SpiceStateRequest(BaseModel):
    target: str
    utc: str
    observer: str = "EARTH"
    frame: str = "J2000"


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
