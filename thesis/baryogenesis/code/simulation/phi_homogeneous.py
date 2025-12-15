import numpy as np
import logging

from model import slow_roll
from integrators.rk4 import RK4Integrator

logger = logging.getLogger(__name__)

def run(args, ic):
    integrator = RK4Integrator(slow_roll.phi_rhs_slowroll)

    header = ["phi"]

    init_dict = ic["initial_state"]
    initial_state = np.array(
        [init_dict[name] for name in header]
    )

    tspan = ic["t_span"]
    N0    = tspan["t0"]
    N_end = tspan["t_end"]
    dN    = tspan["dt"]

    params = ic["parameters"]

    simulation_time = N_end - N0

    t_values, states = integrator.integrate(
        simulation_time,
        dN,
        initial_state,
        params
    )

    return t_values, states, header
