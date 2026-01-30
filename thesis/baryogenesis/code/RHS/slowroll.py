import numpy as np
import logging

logger = logging.getLogger(__name__)

import numpy as np

def phi_slowroll_rhs(N, state, params):
    # unpack state
    phi = state[0]

    # parameters
    kappa = float(params["kappa"])
    b = float(params["b"])

    dphi_dN = -(kappa / b) * np.exp(phi / b)

    return np.array([dphi_dN])