import numpy as np
import logging

from model import harmonic
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
    t_end = tspan["t_end"]
    t0 = tspan["t0"]
    dt = tspan["dt"]
    params = ic["parameters"]

    simulation_time = t_end - t0

    t_values, states = integrator.integrate(simulation_time, dt, initial_state, params)

    return t_values, states, header