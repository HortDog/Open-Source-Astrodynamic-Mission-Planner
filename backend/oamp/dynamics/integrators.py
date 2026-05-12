"""Symplectic integrators for long-arc orbital propagation.

DOP853 (in ``newtonian.propagate_orbit``) is the workhorse — it adapts step
size and is accurate for short-to-medium arcs. For multi-revolution or
multi-month propagations where energy drift matters more than step adaption,
a symplectic scheme is preferable: it bounds the secular energy error and
preserves the symplectic two-form, so apsides do not slowly migrate.

We implement two members of the family:

* ``leapfrog_step`` — 2nd-order velocity-Verlet ("kick-drift-kick"). Symplectic
  for separable Hamiltonians H = T(v) + V(r). Drag and SRP depend on v so the
  scheme is no longer strictly symplectic when those are enabled, but it
  remains accurate and stable.

* ``yoshida4_step`` — 4th-order composition of three Verlet steps with
  Yoshida's coefficients (Yoshida 1990). Roughly 3× the cost of Verlet per
  step for ~10× longer steps before energy drift sets in.

Both work with the same RHS-style acceleration function:

    a(t, r, v) -> np.ndarray
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from oamp.bodies import EARTH, Body
from oamp.dynamics.newtonian import Maneuver, TwoBodyState, ric_to_inertial
from oamp.dynamics.perturbations import Perturbation, zonal_harmonics

AccelFn = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


def _make_accel(mu: float, perturbation: Perturbation | None) -> AccelFn:
    """Wrap the two-body acceleration + optional perturbation into a single fn."""
    if perturbation is None:

        def _a(_t: float, r: np.ndarray, _v: np.ndarray) -> np.ndarray:
            rn = float(np.linalg.norm(r))
            return -mu * r / (rn * rn * rn)
    else:

        def _a(t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
            rn = float(np.linalg.norm(r))
            return -mu * r / (rn * rn * rn) + perturbation(t, r, v)

    return _a


def leapfrog_step(
    t: float,
    r: np.ndarray,
    v: np.ndarray,
    h: float,
    accel: AccelFn,
) -> tuple[np.ndarray, np.ndarray]:
    """One step of velocity-Verlet (kick-drift-kick).

    v_half = v + (h/2) a(t, r, v)
    r_new  = r + h v_half
    v_new  = v_half + (h/2) a(t + h, r_new, v_half)
    """
    a0 = accel(t, r, v)
    v_half = v + 0.5 * h * a0
    r_new = r + h * v_half
    a1 = accel(t + h, r_new, v_half)
    v_new = v_half + 0.5 * h * a1
    return r_new, v_new


# Yoshida 4th-order coefficients (Yoshida 1990, eqn 16). One step = three
# composed Verlet sub-steps of size w0, w1, w0.
_Y4_X1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_Y4_X0 = -(2.0 ** (1.0 / 3.0)) * _Y4_X1


def yoshida4_step(
    t: float,
    r: np.ndarray,
    v: np.ndarray,
    h: float,
    accel: AccelFn,
) -> tuple[np.ndarray, np.ndarray]:
    """One step of Yoshida's 4th-order symplectic integrator."""
    r1, v1 = leapfrog_step(t, r, v, _Y4_X1 * h, accel)
    r2, v2 = leapfrog_step(t + _Y4_X1 * h, r1, v1, _Y4_X0 * h, accel)
    r3, v3 = leapfrog_step(t + (_Y4_X1 + _Y4_X0) * h, r2, v2, _Y4_X1 * h, accel)
    return r3, v3


_STEPPERS: dict[
    str, Callable[[float, np.ndarray, np.ndarray, float, AccelFn], tuple[np.ndarray, np.ndarray]]
] = {
    "verlet": leapfrog_step,
    "leapfrog": leapfrog_step,
    "yoshida4": yoshida4_step,
}


def propagate_symplectic(
    state: TwoBodyState,
    duration_s: float,
    steps: int = 200,
    body: Body = EARTH,
    perturbation: Perturbation | None = None,
    j2_enabled: bool = False,
    maneuvers: list[Maneuver] | None = None,
    t0_tdb: float = 0.0,
    method: str = "yoshida4",
    substeps_per_sample: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate via a fixed-step symplectic scheme.

    Unlike DOP853 this does not adapt — accuracy is controlled by
    ``substeps_per_sample`` (how many integrator steps run between recorded
    samples). For typical LEO/GEO arcs ``yoshida4`` with 4 substeps and 200
    samples per orbit conserves energy to ~10⁻¹².

    Manoeuvres split the trajectory at exact epochs (see ``propagate_orbit``).
    """
    if method not in _STEPPERS:
        raise ValueError(f"unknown integrator {method!r}; choose from {list(_STEPPERS)}")
    stepper = _STEPPERS[method]

    if perturbation is None and j2_enabled and body.j2 != 0.0:
        perturbation = zonal_harmonics(body, n_max=2)
    accel = _make_accel(body.mu, perturbation)

    mans = sorted(maneuvers or [], key=lambda m: m.t_offset_s)
    if any(m.t_offset_s > duration_s for m in mans):
        raise ValueError("manoeuvre occurs after the requested propagation end")

    breakpoints = [0.0, *(m.t_offset_s for m in mans), duration_s]
    total = duration_s if duration_s > 0 else 1.0
    arc_samples = [
        max(2, int(round(steps * (breakpoints[i + 1] - breakpoints[i]) / total)))
        for i in range(len(breakpoints) - 1)
    ]

    times_out: list[float] = []
    states_out: list[np.ndarray] = []
    r = np.asarray(state.r, dtype=float)
    v = np.asarray(state.v, dtype=float)

    for arc_i, n_samples in enumerate(arc_samples):
        t_a, t_b = breakpoints[arc_i], breakpoints[arc_i + 1]
        if t_b <= t_a:
            continue
        # Build sample times for this arc.
        sample_times = np.linspace(t_a, t_b, n_samples)

        # The integrator runs internally at substeps_per_sample × the sample rate.
        sample_dt = (t_b - t_a) / max(n_samples - 1, 1)
        h = sample_dt / max(substeps_per_sample, 1)
        t = t_a

        # Record the starting sample (or the post-manoeuvre sample for arcs > 0).
        first_idx = 0 if arc_i == 0 else 1
        if first_idx == 0:
            times_out.append(t)
            states_out.append(np.concatenate((r, v)))

        for i in range(1, n_samples):
            target_t = float(sample_times[i])
            while t + h < target_t - 1e-12:
                r, v = stepper(t0_tdb + t, r, v, h, accel)
                t += h
            # Last partial step lands exactly on the sample.
            remaining = target_t - t
            if remaining > 0:
                r, v = stepper(t0_tdb + t, r, v, remaining, accel)
                t = target_t
            times_out.append(t)
            states_out.append(np.concatenate((r, v)))

        # Apply the manoeuvre at the end of this arc (if any).
        if arc_i < len(mans):
            dv_inertial = ric_to_inertial(r, v, np.asarray(mans[arc_i].dv_ric))
            v = v + dv_inertial

    return np.asarray(times_out, dtype=float), np.vstack(states_out)


def propagate_symplectic_chunked(
    state: TwoBodyState,
    duration_s: float,
    steps: int = 200,
    body: Body = EARTH,
    perturbation: Perturbation | None = None,
    j2_enabled: bool = False,
    maneuvers: list[Maneuver] | None = None,
    t0_tdb: float = 0.0,
    method: str = "yoshida4",
    substeps_per_sample: int = 4,
    chunk_size: int = 200,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Like propagate_symplectic but yields (t_chunk, states_chunk) every chunk_size steps."""
    if method not in _STEPPERS:
        raise ValueError(f"unknown integrator {method!r}; choose from {list(_STEPPERS)}")
    stepper = _STEPPERS[method]

    if perturbation is None and j2_enabled and body.j2 != 0.0:
        perturbation = zonal_harmonics(body, n_max=2)
    accel = _make_accel(body.mu, perturbation)

    mans = sorted(maneuvers or [], key=lambda m: m.t_offset_s)
    if any(m.t_offset_s > duration_s for m in mans):
        raise ValueError("manoeuvre occurs after the requested propagation end")

    breakpoints = [0.0, *(m.t_offset_s for m in mans), duration_s]
    total = duration_s if duration_s > 0 else 1.0
    arc_samples = [
        max(2, int(round(steps * (breakpoints[i + 1] - breakpoints[i]) / total)))
        for i in range(len(breakpoints) - 1)
    ]

    r = np.asarray(state.r, dtype=float)
    v = np.asarray(state.v, dtype=float)
    t_buf: list[float] = []
    s_buf: list[np.ndarray] = []

    for arc_i, n_samples in enumerate(arc_samples):
        t_a, t_b = breakpoints[arc_i], breakpoints[arc_i + 1]
        if t_b <= t_a:
            continue
        sample_times = np.linspace(t_a, t_b, n_samples)
        sample_dt = (t_b - t_a) / max(n_samples - 1, 1)
        h = sample_dt / max(substeps_per_sample, 1)
        t = t_a

        if arc_i == 0:
            t_buf.append(t)
            s_buf.append(np.concatenate((r, v)))

        for i in range(1, n_samples):
            target_t = float(sample_times[i])
            while t + h < target_t - 1e-12:
                r, v = stepper(t0_tdb + t, r, v, h, accel)
                t += h
            remaining = target_t - t
            if remaining > 0:
                r, v = stepper(t0_tdb + t, r, v, remaining, accel)
                t = target_t
            t_buf.append(t)
            s_buf.append(np.concatenate((r, v)))

            if len(t_buf) >= chunk_size:
                yield np.asarray(t_buf, dtype=float), np.vstack(s_buf)
                t_buf = []
                s_buf = []

        if arc_i < len(mans):
            dv_inertial = ric_to_inertial(r, v, np.asarray(mans[arc_i].dv_ric))
            v = v + dv_inertial

    if t_buf:
        yield np.asarray(t_buf, dtype=float), np.vstack(s_buf)
