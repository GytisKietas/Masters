import numpy as np
import logging

from utils import io

from RHS import slowroll
from integrators.rk4 import RK4Integrator

logger = logging.getLogger(__name__)

np.seterr(over='raise', invalid='raise', divide='raise')

def run(args, ic):
    integrator = RK4Integrator(slowroll.phi_slowroll_rhs)

    header = ["N", "phi"]

    init = ic["initial_state"]

    phi0 = float(init["phi"])

    initial_state = np.array([phi0], dtype=float)

    tspan = ic["t_span"]
    params = ic["parameters"]

    def on_save(i, t_values, states):
        logging.info(f"SAVING CHECKPOINT {i}")
        name = args.name + f"CKPT_{i}"
        io.save_checkpoint(name, t_values, states, header)

    def on_end_condition(t_current, state):
        epsilon = (state[1]**2) / 2 #phi_prime^2/2 with mpl=1

        return epsilon >= 1.0

    t_values, states = integrator.integrate(tspan, initial_state, params, on_save)

    metadata = {
        "header" : header,
    }

    return t_values, states, metadata

