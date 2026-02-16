import numpy as np
import logging

from RHS import phi_N

from integrators.rk4 import RK4Integrator

logger = logging.getLogger(__name__)

def run(args, ic):

    # integrator = RK4Integrator(phi.phi_alpha_rhs)
    integrator = RK4Integrator(phi_N.phi_b_rhs)
    header = ["N", "phi", "phi_prime"]

    N_span = ic["N_span"]
    params = ic["parameters"]

    phi_sbp = params["phi_sbp"]

    kappa = params["kappa"]
    # alpha = params["alpha"]
    
    b = params["b"]
    alpha = b**2

    N_end = N_span["N_end"]

    params["N_end"] = N_end

    phi0, phi_prime0 = compute_initial_conditions(kappa, alpha, N_end)

    initial_state = np.array([phi0, phi_prime0], dtype=float)

    def on_end_condition_upper_bound(t_current, state):
        return state[0] < phi_sbp #If phi comes from above phi_sbp then condition triggers when phi dips below
    
    def on_end_condition_lower_bound(t_current, state):
        return state[0] > phi_sbp #If phi comes from below phi_sbp so condition triggers when phi exceeds phi_sbp

    on_end_condition = on_end_condition_upper_bound if phi0 > 0 else on_end_condition_lower_bound

    N_values, states = integrator.integrate_conditional(tspan=N_span, initial_state=initial_state, params=params, on_end_condition=on_end_condition)

    metadata = {
        "header" : header,
        "params" : params,
    }

    return N_values, states, metadata

def compute_initial_conditions(kappa, alpha, N_end):
    phi0 = -np.sqrt(alpha) * np.log(
        kappa / np.sqrt(2 * alpha) + (kappa / alpha) * N_end
    )

    phi0_prime = 1.0 / (N_end / np.sqrt(alpha) + 1.0 / np.sqrt(2))

    return phi0, phi0_prime
