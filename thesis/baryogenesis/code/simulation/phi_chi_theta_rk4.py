import numpy as np
import logging

from integrators.rk4 import RK4Integrator
from model.phi_chi_theta import coupled_rhs

logger = logging.getLogger(__name__)

def run(args):
    logger.info("Coupled φ–χ–θ RK4 simulation running...")

    num_modes = 2

    # initial conditions
    phi0      = 1.0
    phi_dot0  = 0.0
    chi0      = np.array([0.1, 0.1])
    chi_dot0  = np.array([0.0, 0.0])
    theta0    = np.array([1.0, 0.5])
    theta_dot0 = np.array([0.0, 0.0])

    initial_state = np.concatenate([
        np.array([phi_dot0, phi0]),
        chi_dot0, chi0,
        theta_dot0, theta0
    ])

    params = {
        "num_modes": num_modes,
        "hubble": 1.0,
        "alpha": 3.0,
        "k": 6.0,

        "g": 1.0,
        "phi_sbp": 0.0,
        "lambda": 0.1,
        "f": 1.0,

        "kappa": 1.0,
        "V0": 1.0,
        "b": 1.0,
        "M_pl": 1.0,

        # "weights": weights_array,
    }

    integrator = RK4Integrator(coupled_rhs)
    t_values, states = integrator.integrate(10.0, 0.01, initial_state, params)

    return t_values, states
