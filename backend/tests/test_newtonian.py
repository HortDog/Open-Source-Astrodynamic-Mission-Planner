import math

import numpy as np
from oamp.dynamics.newtonian import TwoBodyState, propagate_two_body

MU_EARTH = 3.986004418e14


def test_circular_leo_period_and_energy_conservation():
    r0 = 7_000_000.0
    v0 = math.sqrt(MU_EARTH / r0)
    period = 2 * math.pi * math.sqrt(r0**3 / MU_EARTH)

    state = TwoBodyState(r=(r0, 0.0, 0.0), v=(0.0, v0, 0.0))
    _, ys = propagate_two_body(state, period, steps=400, mu=MU_EARTH)

    # Returns to start within tight tolerance.
    np.testing.assert_allclose(ys[-1, :3], ys[0, :3], rtol=0, atol=1.0)

    # Specific orbital energy is conserved.
    def energy(y: np.ndarray) -> float:
        r = float(np.linalg.norm(y[:3]))
        v2 = float(np.dot(y[3:], y[3:]))
        return 0.5 * v2 - MU_EARTH / r

    e0 = energy(ys[0])
    eN = energy(ys[-1])
    assert abs(eN - e0) / abs(e0) < 1e-9
