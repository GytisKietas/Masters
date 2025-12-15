import numpy as np
import logging

from integrators.rk4 import RK4Integrator
from model.phi_chi_theta import coupled_rhs

logger = logging.getLogger(__name__)

def bunch_davies_ic(k_array, phi, params, a0=1.0,
                    include_hubble=False, real_output=False):
    g = params["g"]
    lam = params["lambda"]
    f = params["f"]

    omega2 = k_array**2 + g**2 * phi**2 - lam * f**2
    omega = np.sqrt(omega2)

    # Hubble term if requested
    if include_hubble:
        M_pl = params.get("M_pl", 1.0)
        v = params.get("v", None)
        if v is None:
            raise ValueError("include_hubble=True but 'v' not found in params.")
        H = np.abs(v) / (np.sqrt(6.0) * M_pl)
    else:
        H = 0.0

    # Bunch–Davies
    chi = 1.0 / (a0 * np.sqrt(2.0 * omega))

    chi_dot = (-H - 1j * omega) * chi

    if real_output:
        chi = chi.real
        chi_dot = chi_dot.real

    return chi, chi_dot


def run(args, ic):
    logger.info("Coupled phi-chi simplified RK4 simulation running...")

    num_modes = ic["num_modes"]

    header = ["phi_dot", "phi"]
    header.extend([f"chi_dot_{k}" for k in range(num_modes)])
    header.extend([f"chi_{k}" for k in range(num_modes)])

    init = ic["initial_state"]
    phi_vec = np.array(init["phi"], dtype=float)
    phi_dot0, phi0 = phi_vec


    params = ic["parameters"]
    params["num_modes"] = num_modes

    k_min = params["k_min"]
    k_max = params["k_max"]
    k_array = np.linspace(k_min, k_max, num_modes)

    a0 = params.get("a0", 1.0)

    params.setdefault("v", phi_dot0)

    chi_vec, chi_dot_vec = bunch_davies_ic(
        k_array,
        phi0,
        params,
        a0=a0,
        include_hubble=True,   # set True for the -H term
        real_output=True
    )

    initial_state = np.concatenate([phi_vec, chi_dot_vec, chi_vec])

    tspan = ic["t_span"]
    t0    = tspan["t0"]
    tf    = tspan["t_end"]
    dt    = tspan["dt"]
    simulation_time = tf - t0

    integrator = RK4Integrator(coupled_rhs)
    t_values, states = integrator.integrate(
        simulation_time, dt, initial_state, params
    )

    return t_values, states, header
