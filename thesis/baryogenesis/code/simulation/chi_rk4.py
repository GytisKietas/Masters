import numpy as np
import logging

from model import chi
from integrators.rk4 import RK4Integrator

logger = logging.getLogger(__name__)

def run(args):
    logger.info("Theta RK4 simulation running...")

    integrator = RK4Integrator(chi.chi_rhs)

    initial_state = np.array([0.1, 1.0, 1.0, 0.5])
    params = {
        "num_modes" : 2,
        "chi_squared_avg" : 2.1,
        "theta_dot_sq_avg" : 3.1,
        "hubble" : 1.0,
        "k" : 6.0,
        "alpha" : 3.0,
        "g" : 8.0,
        "grad_theta_sq_avg" : 9.1,
        "phi_sbp" : 1.1,
        "lambda" : 5.0,
        "f" : 4.2,
        "phi" : 3.0,
        "a" : 1.0
    }

    results = integrator.integrate(10, 0.01, initial_state, params)

    return results