"""Open-loop launch simulator (MVP).

A 2D ascent in the launch site's local plane:
- spherical Earth, no rotation (ignore the ~465 m/s the launch site gets
  from Earth's spin; budget extra Δv to compensate)
- exponential atmosphere ρ(h) = ρ₀ exp(-h / H)
- constant thrust at constant Isp until propellant exhausted
- thrust direction follows a pre-programmed pitch schedule: vertical until
  `pitch_start_alt_m`, then linearly ramping toward `pitch_target_deg`
  (measured from local vertical) by the time we reach `pitch_target_alt_m`

A real launcher would close the loop with PEG-style guidance once it's out
of the atmosphere; for the MVP, the open-loop program is enough to put the
vehicle on a stable orbit.

Output: a single-stage trajectory in ECI from t=0 (liftoff) integrated until
either engine cutoff + a short coast, or until the vehicle re-enters /
crashes. The result is suitable for the WebGPU visualizer.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel
from scipy.integrate import solve_ivp

from oamp.bodies import EARTH

# Earth atmosphere — crude exponential model; good enough through ~80 km.
RHO_0 = 1.225        # kg/m^3 sea-level
SCALE_HEIGHT = 8500  # m
G0 = 9.80665         # standard gravity, used for Isp -> exhaust velocity


class Vehicle(BaseModel):
    dry_mass_kg: float
    prop_mass_kg: float
    thrust_n: float
    isp_s: float
    drag_area_m2: float = 10.0
    drag_cd: float = 0.30


class LaunchConfig(BaseModel):
    vehicle: Vehicle
    # Below pitch_start_alt_m, thrust is purely vertical. Between that and
    # pitch_target_alt_m, thrust pitches linearly from 0° to pitch_target_deg
    # (measured from local vertical).
    pitch_start_alt_m: float = 1500.0
    pitch_target_alt_m: float = 100_000.0
    pitch_target_deg: float = 88.0
    coast_after_burnout_s: float = 200.0


class LaunchResult(BaseModel):
    t: list[float]
    states: list[list[float]]  # [x, y, z, vx, vy, vz, mass] per row
    burnout_index: int           # index in t/states where main engine cut off
    circularization_index: int   # index where the apoapsis Δv was applied
    burnout_time_s: float
    circularization_dv_m_s: float
    final_apoapsis_km: float
    final_periapsis_km: float
    final_speed_m_s: float


def _atmospheric_density(altitude_m: float) -> float:
    if altitude_m < 0:
        return RHO_0
    if altitude_m > 1.5e5:
        return 0.0
    return RHO_0 * math.exp(-altitude_m / SCALE_HEIGHT)


def _orbit_apsides(r_vec: np.ndarray, v_vec: np.ndarray, mu: float) -> tuple[float, float]:
    """Return (apoapsis, periapsis) in m from a Cartesian state."""
    r = float(np.linalg.norm(r_vec))
    v2 = float(np.dot(v_vec, v_vec))
    energy = 0.5 * v2 - mu / r
    if energy >= 0:
        return float("inf"), float("inf")
    a = -mu / (2 * energy)
    h_vec = np.cross(r_vec, v_vec)
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = float(np.linalg.norm(e_vec))
    return a * (1 + e), a * (1 - e)


def _kepler_rhs(_t: float, y: np.ndarray, mu: float) -> np.ndarray:
    r = y[:3]
    v = y[3:6]
    r_norm = float(np.linalg.norm(r))
    a = -mu * r / r_norm**3
    return np.concatenate((v, a, [0.0]))  # mass constant during coast


def _time_to_apoapsis(r_vec: np.ndarray, v_vec: np.ndarray, mu: float) -> float:
    """Closed-form time from current state to apoapsis on the bound orbit."""
    r = float(np.linalg.norm(r_vec))
    v2 = float(np.dot(v_vec, v_vec))
    energy = 0.5 * v2 - mu / r
    if energy >= 0:
        return math.inf
    a = -mu / (2 * energy)
    h_vec = np.cross(r_vec, v_vec)
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = float(np.linalg.norm(e_vec))
    if e < 1e-9:
        return 0.0  # already circular — pick "now" as the apoapsis
    # True anomaly from r_vec, e_vec (sign from radial velocity).
    cos_nu = float(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
    nu = math.acos(cos_nu)
    if float(np.dot(r_vec, v_vec)) < 0:
        nu = 2 * math.pi - nu
    # Eccentric anomaly, then mean anomaly.
    E = 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2), math.sqrt(1 + e) * math.cos(nu / 2))
    M = E - e * math.sin(E)
    n = math.sqrt(mu / a**3)  # mean motion
    # Apoapsis is at M = π.
    dt = (math.pi - M) / n
    if dt < 0:
        dt += 2 * math.pi / n
    return dt


def simulate_launch(config: LaunchConfig, max_time_s: float = 1500.0) -> LaunchResult:
    """Simulate liftoff in the equatorial plane, launching due east.

    Pad placed at (R_E, 0, 0). Initial velocity zero (Earth rotation ignored).
    The launch plane is XY; Z stays zero throughout. After the open-loop
    pitch program raises the apoapsis, the simulator coasts to apoapsis,
    applies an instantaneous Δv to circularize, and propagates one full
    orbit so the demo shows ascent → coast → insertion → orbit in a single
    trajectory.
    """
    body = EARTH
    R = body.radius
    mu = body.mu

    v_e = config.vehicle.isp_s * G0  # exhaust velocity, m/s
    m_dot = config.vehicle.thrust_n / v_e  # propellant consumption, kg/s
    initial_mass = config.vehicle.dry_mass_kg + config.vehicle.prop_mass_kg
    burnout_t = config.vehicle.prop_mass_kg / m_dot if m_dot > 0 else math.inf
    end_t = min(max_time_s, burnout_t + config.coast_after_burnout_s)

    pitch_target_rad = math.radians(config.pitch_target_deg)
    h_start = config.pitch_start_alt_m
    h_target = config.pitch_target_alt_m

    def pitch_from_altitude(altitude: float) -> float:
        """Fraction-based linear pitch program, clamped at the target."""
        if altitude <= h_start:
            return 0.0
        if altitude >= h_target:
            return pitch_target_rad
        return pitch_target_rad * (altitude - h_start) / (h_target - h_start)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        r_vec = y[:3]
        v_vec = y[3:6]
        m = y[6]

        r_norm = float(np.linalg.norm(r_vec))
        altitude = r_norm - R
        speed = float(np.linalg.norm(v_vec))
        radial_hat = r_vec / r_norm

        # --- thrust ---
        if t < burnout_t and m > config.vehicle.dry_mass_kg:
            pitch = pitch_from_altitude(altitude)
            # In the equatorial launch plane (Z=0), east is the radial vector
            # rotated 90° CCW about +Z.
            east_hat = np.array([-radial_hat[1], radial_hat[0], 0.0])
            thrust_dir = math.cos(pitch) * radial_hat + math.sin(pitch) * east_hat
            thrust_acc = thrust_dir * (config.vehicle.thrust_n / m)
            mass_dot = -m_dot
        else:
            thrust_acc = np.zeros(3)
            mass_dot = 0.0

        # --- gravity ---
        grav_acc = -mu * r_vec / r_norm**3

        # --- drag (relative to atmosphere; we ignore Earth rotation -> v_rel = v) ---
        rho = _atmospheric_density(altitude)
        if rho > 0 and speed > 0:
            cd_a = config.vehicle.drag_cd * config.vehicle.drag_area_m2
            drag_force = 0.5 * rho * speed * speed * cd_a
            drag_acc = -drag_force / m * (v_vec / speed)
        else:
            drag_acc = np.zeros(3)

        return np.concatenate((v_vec, thrust_acc + grav_acc + drag_acc, [mass_dot]))

    def crashed(_t: float, y: np.ndarray) -> float:
        # Stop on impact: r drops below body radius. Skip the first second so
        # the integrator doesn't trip on the initial r == R state.
        r_norm = float(np.linalg.norm(y[:3]))
        return r_norm - R - 1.0

    crashed.terminal = True  # type: ignore[attr-defined]
    crashed.direction = -1   # type: ignore[attr-defined]

    y0 = np.array([R, 0.0, 0.0, 0.0, 0.0, 0.0, initial_mass])
    t_eval = np.linspace(0.0, end_t, max(200, int(end_t / 2)))
    sol = solve_ivp(
        rhs,
        (0.0, end_t),
        y0,
        t_eval=t_eval,
        events=crashed,
        method="DOP853",
        rtol=1e-8,
        atol=1e-6,
        first_step=0.05,
        max_step=2.0,
    )

    ascent_states = sol.y.T   # (N, 7)
    ascent_times = sol.t

    # Find the index nearest to burnout time.
    burnout_idx = int(np.argmin(np.abs(ascent_times - min(burnout_t, ascent_times[-1]))))

    # Coast to apoapsis on a Kepler arc (no drag, no thrust).
    coast_start_t = float(ascent_times[-1])
    coast_start_state = ascent_states[-1].copy()
    apo_radius, _ = _orbit_apsides(coast_start_state[:3], coast_start_state[3:6], mu)

    if not math.isfinite(apo_radius) or apo_radius <= R:
        # Sub-orbital — return what we have without circularization.
        final_r = ascent_states[-1, :3]
        final_v = ascent_states[-1, 3:6]
        apo, peri = _orbit_apsides(final_r, final_v, mu)
        return LaunchResult(
            t=[float(x) for x in ascent_times],
            states=[[float(v) for v in row] for row in ascent_states],
            burnout_index=burnout_idx,
            circularization_index=len(ascent_times) - 1,
            burnout_time_s=float(burnout_t),
            circularization_dv_m_s=0.0,
            final_apoapsis_km=(apo - R) / 1000 if math.isfinite(apo) else float("inf"),
            final_periapsis_km=(peri - R) / 1000 if math.isfinite(peri) else float("inf"),
            final_speed_m_s=float(np.linalg.norm(final_v)),
        )

    dt_to_apo = _time_to_apoapsis(coast_start_state[:3], coast_start_state[3:6], mu)
    coast_t_eval = np.linspace(0.0, dt_to_apo, max(60, int(dt_to_apo / 5.0)))
    coast_sol = solve_ivp(
        _kepler_rhs, (0.0, dt_to_apo), coast_start_state,
        t_eval=coast_t_eval, args=(mu,), method="DOP853", rtol=1e-10, atol=1e-12,
    )
    coast_states = coast_sol.y.T
    coast_times = coast_sol.t + coast_start_t

    # Circularize at apoapsis: replace velocity magnitude with v_circ along
    # the same direction (which is purely tangential at apoapsis).
    apo_state = coast_states[-1].copy()
    r_apo = apo_state[:3]
    v_apo = apo_state[3:6]
    v_apo_mag = float(np.linalg.norm(v_apo))
    v_circ = math.sqrt(mu / float(np.linalg.norm(r_apo)))
    dv = v_circ - v_apo_mag
    if v_apo_mag > 0:
        apo_state[3:6] = v_apo * (v_circ / v_apo_mag)

    # Propagate one full circular orbit.
    period = 2 * math.pi * math.sqrt(float(np.linalg.norm(r_apo)) ** 3 / mu)
    orbit_t_eval = np.linspace(0.0, period, 360)
    orbit_sol = solve_ivp(
        _kepler_rhs, (0.0, period), apo_state,
        t_eval=orbit_t_eval, args=(mu,), method="DOP853", rtol=1e-10, atol=1e-12,
    )
    orbit_states = orbit_sol.y.T
    orbit_times = orbit_sol.t + coast_times[-1]

    # Concatenate, dropping duplicate boundary samples.
    times = np.concatenate([ascent_times, coast_times[1:], orbit_times[1:]])
    states = np.vstack([ascent_states, coast_states[1:], orbit_states[1:]])
    circ_idx = len(ascent_times) + len(coast_times) - 2

    final_r = states[-1, :3]
    final_v = states[-1, 3:6]
    apo, peri = _orbit_apsides(final_r, final_v, mu)
    return LaunchResult(
        t=[float(x) for x in times],
        states=[[float(v) for v in row] for row in states],
        burnout_index=burnout_idx,
        circularization_index=circ_idx,
        burnout_time_s=float(burnout_t),
        circularization_dv_m_s=float(dv),
        final_apoapsis_km=(apo - R) / 1000 if math.isfinite(apo) else float("inf"),
        final_periapsis_km=(peri - R) / 1000 if math.isfinite(peri) else float("inf"),
        final_speed_m_s=float(np.linalg.norm(final_v)),
    )


def default_falcon9_like() -> LaunchConfig:
    """A toy Falcon-9-ish single-stage profile that just makes orbit.

    Numbers are tuned so the demo reaches a stable LEO without needing
    staging — they're not a real F9 model.
    """
    return LaunchConfig(
        vehicle=Vehicle(
            dry_mass_kg=12_000,
            prop_mass_kg=200_000,
            thrust_n=8_200_000,   # ~9 Merlin equivalents
            isp_s=320,
            drag_area_m2=10.5,
            drag_cd=0.30,
        ),
        pitch_start_alt_m=1500,
        pitch_target_alt_m=20_000,
        pitch_target_deg=89,
        coast_after_burnout_s=200,
    )
