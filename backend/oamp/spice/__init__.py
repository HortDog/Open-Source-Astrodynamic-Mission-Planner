"""SPICE wrapper: lazy kernel loading + state queries in SI units.

SPICE (via SpiceyPy) returns positions in km and velocities in km/s. We
convert to meters at the boundary so the rest of OAMP can stay in SI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

KM_TO_M: Final[float] = 1000.0


_loaded: set[Path] = set()


def kernels_dir() -> Path:
    """Project-relative kernels directory: <repo>/data/kernels."""
    # backend/oamp/spice/__init__.py -> repo root is 3 levels up.
    return Path(__file__).resolve().parents[3] / "data" / "kernels"


def furnsh_dir(directory: Path | None = None) -> list[Path]:
    """Load every .tls / .bsp / .tpc / .bpc / .tf in the given directory.

    Idempotent — already-loaded kernels are skipped. Returns the list of
    kernels actually loaded by this call.
    """
    import spiceypy as sp

    d = directory or kernels_dir()
    if not d.is_dir():
        raise FileNotFoundError(
            f"SPICE kernels directory not found: {d}. "
            "Run `pixi run kernels` to fetch them."
        )

    loaded_now: list[Path] = []
    for ext in (".tls", ".bsp", ".tpc", ".bpc", ".tf"):
        for path in sorted(d.glob(f"*{ext}")):
            if path in _loaded:
                continue
            sp.furnsh(str(path))
            _loaded.add(path)
            loaded_now.append(path)
    return loaded_now


def body_state(
    target: str,
    et: float,
    observer: str = "EARTH",
    frame: str = "J2000",
    aberration: str = "NONE",
) -> tuple[np.ndarray, np.ndarray]:
    """Position (m) and velocity (m/s) of `target` relative to `observer`.

    `et` is TDB seconds since J2000 (== `oamp.timescales.utc_iso_to_et(...)`).
    """
    import spiceypy as sp

    state, _light_time = sp.spkezr(target, et, frame, aberration, observer)
    arr = np.asarray(state, dtype=float) * KM_TO_M
    return arr[:3], arr[3:]


def body_position(
    target: str,
    et: float,
    observer: str = "EARTH",
    frame: str = "J2000",
) -> np.ndarray:
    return body_state(target, et, observer, frame)[0]
