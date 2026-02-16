import numpy as np

def unpack_state(state, params):
    state = np.asarray(state, dtype=float)
    n_k = int(params["n_k"])

    # Layout:
    # [phi, chi(0..n_k-1), phi_dot, chi_dot(0..n_k-1), a]
    phi = state[0]
    chi = state[1 : 1 + n_k]

    phi_dot = state[1 + n_k]
    chi_dot = state[2 + n_k : 2 + 2*n_k]

    a = state[2 + 2*n_k]
    return phi, phi_dot, chi, chi_dot, a


def V_func(phi, V0, kappa, b):
    return V0 * np.exp(-kappa * np.exp(b * phi))

def H_func(phi, phi_dot, V0, kappa, b):
    H2 = (8.0 * np.pi / 3.0) * (0.5 * phi_dot**2 + V_func(phi, V0, kappa, b))
    return np.sqrt(H2)

def dV_dphi_func(phi, b, V0, kappa):
    exp_term = np.exp(b * phi)
    V = V0 * np.exp(-kappa * exp_term)
    return -(b * kappa) * exp_term * V

def omega2_func(phi, k, a, g, phi_sbp):
    return (k**2) / (a**2) + g**2 * (phi - phi_sbp)**2


def phi_chi_cosmic_time_rhs(t, state, params):
    g = float(params["g"])
    phi_sbp = float(params["phi_sbp"])
    k_min = float(params["k_min"])
    k_max = float(params["k_max"])
    n_k = int(params["n_k"])
    k = np.linspace(k_min, k_max, n_k)

    V0 = float(params["V0"])
    kappa = float(params["kappa"])
    b = float(params["b"])

    phi, phi_dot, chi, chi_dot, a = unpack_state(state, params)

    H = H_func(phi, phi_dot, V0, kappa, b)
    dV_dphi = dV_dphi_func(phi, b, V0, kappa)
    omega2 = omega2_func(phi, k, a, g, phi_sbp)

    phi_ddot = -3.0 * H * phi_dot - dV_dphi
    chi_ddot = -3.0 * H * chi_dot - omega2 * chi

    a_dot = H * a

    return np.concatenate((
        np.array([phi_dot], dtype=float),
        chi_dot.astype(float),
        np.array([phi_ddot], dtype=float),
        chi_ddot.astype(float),
        np.array([a_dot], dtype=float),
    ))
