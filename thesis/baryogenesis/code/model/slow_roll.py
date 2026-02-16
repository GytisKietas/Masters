import numpy as np
import logging

logger = logging.getLogger(__name__)

def phi_rhs_slowroll(N_current, state, params):
    kappa = params.get("kappa", 0.0)
    b     = params.get("b", 1.0)

    phi = state[0]

    exponent = phi / b
    W_phi    = - (kappa / b) * np.exp(exponent)

    dphi_dN  = W_phi

    return np.array([dphi_dN])
