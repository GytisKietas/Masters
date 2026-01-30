import numpy as np
import logging

from RHS import harmonic
from integrators.rk4 import RK4Integrator

logger = logging.getLogger(__name__)

def run(args, ic):
    integrator = RK4Integrator(harmonic.coupled_oscillators_rhs)

    header = ["x1", "x2", "v1", "v2"]

    init_dict = ic["initial_state"]
    initial_state = np.array(
        [init_dict[name] for name in header],
        dtype=float
    )

    tspan = ic["t_span"]
    params = ic["parameters"]

    t_values, states = integrator.integrate(tspan, initial_state, params)

    return t_values, states, header