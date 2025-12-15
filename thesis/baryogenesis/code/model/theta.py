import numpy as np
import logging

logger = logging.getLogger(__name__)

def theta_rhs(t_current, state, params):
    num_modes = params.get("num_modes", 1)
    chi_chidot_avg = params.get("chi_chidot_avg", 0)
    chi_squared_avg = params.get("chi_squared_avg", 0)
    hubble = params.get("hubble", 0)
    k = params.get("k", 0)
    alpha = params.get("alpha", 0) #I think this is alpha, though Apostolos wrote it as 'a'

    theta = state[num_modes : ]
    theta_dot = state[ : num_modes]

    dTheta_dt = theta_dot

    dThetaDot_dt = - (2 * (chi_chidot_avg/chi_squared_avg) + 3 * hubble) * theta_dot - (k**2 / alpha**2) * theta
    
    return np.concatenate([dTheta_dt, dThetaDot_dt])