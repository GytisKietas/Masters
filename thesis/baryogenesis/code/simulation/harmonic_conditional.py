import numpy as np
import logging

from RHS import harmonic
from integrators.rk4 import RK4Integrator

logger = logging.getLogger(__name__)

def run(args, ic):
    integrator = RK4Integrator(harmonic.harmonic_rhs)

    header = ["t", "x", "v"]

    init = ic["initial_state"]
    x0 = init["x0"]
    v0 = init["v0"]
    initial_state = np.array( [x0, v0], dtype=float)

    tspan = ic["t_span"]
    params = ic["parameters"]
    condition = ic["condition"]

    x_end = condition["end_value"]

    def on_end_condition(t_current, state):
        return state[0] <= x_end

    t_values, states = integrator.integrate_conditional(tspan=tspan, initial_state=initial_state, params=params, on_end_condition=on_end_condition)

    metadata = {
        "header" : header,
    }

    return t_values, states, metadata