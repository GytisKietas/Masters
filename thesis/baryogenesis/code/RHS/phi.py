import numpy as np
import logging

from integrators.rk4 import RK4Integrator

logger = logging.getLogger(__name__)

def phi_rhs(N, state, params):
    phi, phi_prime = state

    kappa = float(params["kappa"])
    b = float(params["b"])

    dphi_dN = phi_prime

    dphi_prime_dN = (
        (0.5 * phi_prime**2 - 3.0) * phi_prime
        + (3.0 - 0.5 * phi_prime**2) * (kappa / b) * np.exp(phi / b)
    )

    return np.array([dphi_dN, dphi_prime_dN], dtype=np.float64)
