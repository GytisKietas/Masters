import numpy as np
import logging

logger = logging.getLogger(__name__)

def chi_rhs(t_current, state, params):
    num_modes = params.get("num_modes", 1)

    hubble = params.get("hubble", 0)
    k = params.get("k", 0)
    a = params.get("alpha", 1.0)

    g = params.get("g", 0)
    phi = params.get("phi", 0)
    phi_sbp = params.get("phi_sbp", 0)
    lam = params.get("lambda", 0)
    f = params.get("f", 0)

    chi_sq_avg = params.get("chi_squared_avg", 0)
    theta_dot_sq_avg = params.get("theta_dot_sq_avg", 0)
    grad_theta_sq_avg = params.get("grad_theta_sq_avg", 0)

    chi_dot = state[:num_modes]
    chi     = state[num_modes:]

    dChi_dt = chi_dot

    #Effective mass
    M_eff_sq = (
        (k**2) / (a**2)
        + g**2 * (phi - phi_sbp)**2
        - lam * f**2
        + 3 * lam * chi_sq_avg
        + theta_dot_sq_avg
        + (1 / a**2) * grad_theta_sq_avg
    )

    dChiDot_dt = -3 * hubble * chi_dot - M_eff_sq * chi

    return np.concatenate([dChi_dt, dChiDot_dt])
