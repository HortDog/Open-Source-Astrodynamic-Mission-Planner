from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from oamp import __version__
from oamp.bodies import EARTH
from oamp.dynamics.launch import LaunchConfig, default_falcon9_like, simulate_launch
from oamp.dynamics.newtonian import TwoBodyState, propagate_orbit


class HealthResponse(BaseModel):
    status: str
    version: str


class PropagateRequest(BaseModel):
    state: TwoBodyState
    duration_s: float
    steps: Annotated[int, Field(ge=2, le=20_000)] = 200
    mu: float = EARTH.mu
    j2_enabled: bool = False
    body_radius: float = EARTH.radius


class SpiceStateRequest(BaseModel):
    target: str
    utc: str  # ISO 8601 UTC string
    observer: str = "EARTH"
    frame: str = "J2000"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Best-effort SPICE bootstrap. Stays optional — endpoints that require
    # SPICE will surface a clear error if kernels aren't available.
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


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.get("/spice/status")
async def spice_status() -> dict:
    return {
        "loaded_kernels": getattr(app.state, "spice_kernels", []),
        "error": getattr(app.state, "spice_error", None),
    }


@app.post("/propagate")
async def propagate(req: PropagateRequest) -> dict:
    body = type(EARTH)(name="custom", mu=req.mu, radius=req.body_radius, j2=EARTH.j2)
    times, states = propagate_orbit(
        req.state,
        req.duration_s,
        req.steps,
        body=body,
        j2_enabled=req.j2_enabled,
    )
    return {"t": times.tolist(), "states": states.tolist()}


@app.post("/launch")
async def launch(config: LaunchConfig | None = None) -> dict:
    cfg = config or default_falcon9_like()
    result = simulate_launch(cfg)
    return result.model_dump()


@app.get("/launch/default-config")
async def launch_default_config() -> dict:
    return default_falcon9_like().model_dump()


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


# Frame: type code (1 byte: 1=trajectory, 2=launch, 3=ack), payload bytes.
WSCommand = Literal["echo", "subscribe_progress"]


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid json"})
                continue
            await websocket.send_json({"echo": payload})
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        return
