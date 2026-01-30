import numpy as np
import logging

from utils import io

from RHS import phi
from integrators.rk4 import RK4Integrator

logger = logging.getLogger(__name__)

np.seterr(over='raise', invalid='raise', divide='raise')

#This is an old version where I integrate with some initial conditions of phi and phi prime that I chose basically randomly. This should be used a reference, but not as actual run
def run(args, ic):
    integrator = RK4Integrator(phi.phi_rhs)
    header = ["N", "phi", "phi_prime"]

    init = ic["initial_state"]

    phi0 = float(init["phi"])
    phi_prime0 = float(init["phi_prime"])

    initial_state = np.array([phi0, phi_prime0], dtype=float)

    tspan = ic["t_span"]
    params = ic["parameters"]

    def on_save(i, t_values, states):
        logging.info(f"SAVING CHECKPOINT {i}")
        name = args.name + f"CKPT_{i}"
        io.save_checkpoint(name, t_values, states, header)

    t_values, states = integrator.integrate(tspan, initial_state, params, on_save)

    return t_values, states, header

