import numpy as np
import logging

from RHS import phi_chi

from integrators.rk4 import RK4Integrator

logger = logging.getLogger(__name__)

def run(args, ic):
    integrator = RK4Integrator(phi_chi.phi_chi_cosmic_time_rhs)

    t_span = ic["t_span"]
    params = ic["parameters"]
    initial_states = ic["initial_states"]

    n_k = int(params["n_k"])

    header = (
    ["t", "phi", "phi_prime"]
    + [f"chi_{i}" for i in range(n_k)]
    + [f"chi_dot_{j}" for j in range(n_k)]
    + ["a"]
    )

    # phi0 = np.array([initial_states["phi0"]])
    # phi_dot0 = np.array([initial_states["phi_dot0"]])

    V0 = params["V0"]
    b = params["b"]
    kappa = params["kappa"]
    g = params["g"]
    phi_sbp = params["phi_sbp"]


    phi0, phi_dot0 = get_ic_phi()
    

    #Temp
    # chi0 = np.full(n_k, initial_states["chi0"])
    # chi_dot0 = np.full(n_k, initial_states["chi_dot0"])


    a0 = np.array([1])
    # a0 = np.array([initial_states["a"]])
    # chi0, chi_dot0 = compute_ic_chi()
    # a0 = compute_ic_a()
    k = generate_k_modes(n_k, 1, phi0, phi_sbp, phi_dot0, V0, b, kappa, g, ir_factor=0.1, uv_factor=5.0)

    chi0 = compute_ic_chi_conformal(phi0, phi_dot0, V0, b, kappa, g, phi_sbp, k).ravel()
    chi_dot0 = compute_ic_chi_dot_conformal(k, 1, g, phi0, phi_sbp, phi_dot0, V0, b, kappa).ravel()



    def on_save(i, t_values, states):
        logging.info("NOT SAVING")

    # initial_state = np.concatenate([phi0, chi0, phi_dot0, chi_dot0, a0], dtype=float)
    initial_state = np.concatenate([phi0, chi0, phi_dot0, chi_dot0, a0]).astype(np.complex128)


    t_values, states = integrator.integrate(tspan=t_span, initial_state=initial_state, params=params, on_save=on_save)

    metadata = {
        "header" : header,
        "params" : params, 
    }

    return t_values, states, metadata



def get_ic_phi():
    #todo: make it read initial conditions from a file with name depending on kappa and alpha (not b) 
    #kappa = 200, b = 1.22
    #1.890700000000012437e+01,-7.140279910970007471e+00,4.245404966369981103e-01
    return np.array([-7.140279910970007471e+00]), np.array([4.245404966369981103e-01])


def compute_ic_chi_cosmic(phi, phi_prime, V0, b, kappa, g, phi_sbp, k):
    H = compute_ic_H(phi, phi_prime, V0, b, kappa)
    H_dot = compute_ic_H_dot(phi, phi_prime, V0, b, kappa)

    omega2 = compute_omega2_k(k, 1, g, phi, phi_sbp)
    Omega2 = omega2 - 1.5 * H_dot - 2.25 * H**2

    Omega = np.sqrt(Omega2)

    return 1.0 / np.sqrt(2.0 * Omega)


def compute_ic_chi_conformal(phi, phi_prime, V0, b, kappa, g, phi_sbp, k):
    H = compute_ic_H(phi, phi_prime, V0, b, kappa)
    H_dot = compute_ic_H_dot(phi, phi_prime, V0, b, kappa)

    omega2 = compute_omega2_k(k, 1, g, phi, phi_sbp)
    Omega2 = omega2 - H_dot - H**2

    Omega = np.sqrt(Omega2)

    return 1.0 / np.sqrt(2.0 * Omega)


def V_func(phi, V0, b, kappa):
    return V0 * np.exp(-kappa * np.exp(phi / b))


def dV_dphi(phi, V0, b, kappa):
    V = V_func(phi, V0, b, kappa)
    return -(kappa / b) * np.exp(phi / b) * V


def compute_ic_H_ddot(phi, phi_prime, V0, b, kappa):
    H = compute_ic_H(phi, phi_prime, V0, b, kappa)
    phi_dot = H * phi_prime
    V_phi = dV_dphi(phi, V0, b, kappa)
    return 3.0 * H * phi_dot**2 + phi_dot * V_phi


def compute_ic_H_dot(phi, phi_prime, V0, b, kappa):
    H = compute_ic_H(phi, phi_prime, V0, b, kappa)
    return -0.5 * H**2 * phi_prime**2


def compute_ic_H(phi, phi_prime, V0, b, kappa):
    denominator = 3.0 - 0.5 * phi_prime**2
    return np.sqrt(V_func(phi, V0, b, kappa) / denominator)


def compute_omega2_k(k, a, g, phi, phi_sbp):
    return (k**2) / (a**2) + g**2 * (phi - phi_sbp)**2


def compute_ic_chi_dot_cosmic(k, a, g, phi, phi_sbp, phi_prime, V0, b, kappa):
    H = compute_ic_H(phi, phi_prime, V0, b, kappa)
    H_dot = compute_ic_H_dot(phi, phi_prime, V0, b, kappa)
    H_ddot = compute_ic_H_ddot(phi, phi_prime, V0, b, kappa)
    phi_dot = H * phi_prime

    omega2 = compute_omega2_k(k, a, g, phi, phi_sbp)

    Omega2 = omega2 - 1.5 * H_dot - 2.25 * H**2
    Omega = np.sqrt(Omega2)

    omega2_dot = (
        -2.0 * H * (k**2 / a**2)
        + 2.0 * g**2 * (phi - phi_sbp) * phi_dot
    )

    Omega2_dot = (
        omega2_dot
        - 1.5 * H_ddot
        - 4.5 * H * H_dot
    )

    Omega_dot = Omega2_dot / (2.0 * Omega)

    chi = compute_ic_chi_cosmic(
        phi, phi_prime, V0, b, kappa, g, phi_sbp, k
    )

    return (-1.5 * H - 0.5 * (Omega_dot / Omega) - 1j * Omega) * chi


def compute_ic_chi_dot_conformal(k, a, g, phi, phi_sbp, phi_prime, V0, b, kappa):

    H = compute_ic_H(phi, phi_prime, V0, b, kappa)
    H_dot = compute_ic_H_dot(phi, phi_prime, V0, b, kappa)
    H_ddot = compute_ic_H_ddot(phi, phi_prime, V0, b, kappa)
    phi_dot = H * phi_prime

    omega2 = compute_omega2_k(k, a, g, phi, phi_sbp)

    Om2 = omega2 - H_dot - H**2
    Om = np.sqrt(Om2)

    omega2_dot = (
        -2.0 * H * (k**2 / a**2)
        + 2.0 * g**2 * (phi - phi_sbp) * phi_dot
    )

    Om2_dot = (
        omega2_dot
        - H_ddot
        - 2.0 * H * H_dot
    )

    Om_dot = Om2_dot / (2.0 * Om)

    chi = compute_ic_chi_conformal(phi, phi_prime, V0, b, kappa, g, phi_sbp, k)

    return (-1.5 * H - 0.5 * (Om_dot / Om) - 1j * Om) * chi



def generate_k_modes(n_k, a, phi, phi_sbp, phi_prime, V0, b, kappa, g, ir_factor=0.1, uv_factor=5.0):

    H = compute_ic_H(phi, phi_prime, V0, b, kappa)

    # IR bound
    p_min = ir_factor * H

    # Effective mass scale
    m_eff = g * abs(phi - phi_sbp)

    p_max = uv_factor * max(m_eff, H)

    k_min = a * p_min
    k_max = a * p_max

    # return np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    return np.logspace(np.log10(k_min), np.log10(k_max), n_k).reshape(-1)
