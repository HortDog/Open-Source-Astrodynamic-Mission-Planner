from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from oamp import __version__
from oamp.dynamics.newtonian import TwoBodyState, propagate_two_body


class HealthResponse(BaseModel):
    status: str
    version: str


class PropagateRequest(BaseModel):
    state: TwoBodyState
    duration_s: float
    steps: int = 200
    mu: float = 3.986004418e14  # Earth, m^3/s^2


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Future: load SPICE kernels, warm Numba caches, init metric cache.
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


@app.post("/propagate")
async def propagate(req: PropagateRequest):
    times, states = propagate_two_body(req.state, req.duration_s, req.steps, req.mu)
    return {"t": times.tolist(), "states": states.tolist()}


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
