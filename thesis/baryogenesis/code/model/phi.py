import numpy as np
import logging

logger = logging.getLogger(__name__)

def phi_rhs(t_current, state, params):
    hubble = params.get("hubble", 0.0)
    kappa  = params.get("kappa", 0.0)
    V_zero     = params.get("V_zero", 0.0)
    b      = params.get("b", 1.0)
    M_pl   = params.get("M_pl", 1.0)
    g      = params.get("g", 0.0)
    chi_sq_avg = params.get("chi_squared_avg", 0.0)
    phi_sbp    = params.get("phi_sbp", 0.0)

    phi_dot = state[0]
    phi     = state[1]

    dphi_dt = phi_dot

    exponent = phi / (b * M_pl)
    exp_factor = np.exp(exponent - kappa * np.exp(exponent))

    dphi_dot_dt = (
        -3.0 * hubble * phi_dot
        + (kappa * V_zero / (b * M_pl)) * exp_factor
        - g**2 * chi_sq_avg * (phi - phi_sbp)
    )

    return np.concatenate([dphi_dt, dphi_dot_dt])
