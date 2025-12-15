import numpy as np
import logging

from model import phi
from integrators.rk4 import RK4Integrator

logger = logging.getLogger(__name__)

def run(args):
    logger.info("Theta RK4 simulation running...")

    integrator = RK4Integrator(phi.phi_rhs)

    initial_state = np.array([0.1, 1.0])
    params = {
        "hubble" : 1.0,
        "kappa" : 1.0,
        "V_zero" : 5.0,
        "b" : 3.1,
        "M_pl" : 1.0,
        "g" : 8.0,
        "chi_squared_avg" : 9.1,
        "phi_sbp" : 1.1
    }

    results = integrator.integrate(10, 0.01, initial_state, params)

    return results