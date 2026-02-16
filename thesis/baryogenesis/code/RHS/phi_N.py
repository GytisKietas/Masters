import numpy as np

def phi_b_rhs(N, state, params):
    phi, phi_prime = state

    kappa = float(params["kappa"])
    b = float(params["b"])

    dphi_dN = phi_prime

    dphi_prime_dN = (
        (0.5 * phi_prime**2 - 3.0) * phi_prime
        + (3.0 - 0.5 * phi_prime**2) * (kappa / b) * np.exp(phi / b)
    )

    return np.array([dphi_dN, dphi_prime_dN], dtype=np.float64)


def phi_alpha_rhs(N, state, params):
    phi, phi_prime = state

    kappa = float(params["kappa"])
    alpha = float(params["alpha"])

    dphi_dN = phi_prime

    dphi_prime_dN = (
        (0.5 * phi_prime**2 - 3.0) * phi_prime
        + (3.0 - 0.5 * phi_prime**2) * (kappa / np.sqrt(alpha)) * np.exp(phi / np.sqrt(alpha))
    )

    return np.array([dphi_dN, dphi_prime_dN], dtype=np.float64)