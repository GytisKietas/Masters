import numpy as np
import logging

logger = logging.getLogger(__name__)

def coupled_rhs(t_current, state, params):
    """
    RHS for the coupled system (phi, chi_k, theta_k).
    Layout of state:
        [phi_dot, phi,
         chi_dot[0..N-1], chi[0..N-1],
         theta_dot[0..N-1], theta[0..N-1]]
    """
    num_modes = params.get("num_modes", 1)

    # --- unpack the state ---
    idx = 0
    phi_dot = state[idx]; idx += 1
    phi     = state[idx]; idx += 1

    chi_dot = state[idx : idx + num_modes]; idx += num_modes
    chi     = state[idx : idx + num_modes]; idx += num_modes

    theta_dot = state[idx : idx + num_modes]; idx += num_modes
    theta     = state[idx : idx + num_modes]

    # --- parameters ---
    hubble = params.get("hubble", 0.0)      
    a      = params.get("alpha", 1.0)       
    k      = params.get("k", 0.0)           
    g      = params.get("g", 0.0)
    phi_sbp = params.get("phi_sbp", 0.0)
    lam    = params.get("lambda", 0.0)
    f      = params.get("f", 0.0)

    kappa  = params.get("kappa", 0.0)
    V_zero     = params.get("V_zero", 0.0)
    b      = params.get("b", 1.0)
    M_pl   = params.get("M_pl", 1.0)


    grad_theta_sq_avg = params.get("grad_theta_sq_avg", 0.0)

    # weights for k-integrals
    weights = params.get("weights", None)
    if weights is None:
        chi_sq_avg       = np.mean(chi**2)
        chi_chidot_avg   = np.mean(chi * chi_dot)
        theta_dot_sq_avg = np.mean(theta_dot**2)
    else:
        chi_sq_avg       = np.sum(weights * chi**2)
        chi_chidot_avg   = np.sum(weights * chi * chi_dot)
        theta_dot_sq_avg = np.sum(weights * theta_dot**2)

    # ------------------------------------------------------------------
    dphi_dt = phi_dot

    exponent   = phi / (b * M_pl)
    exp_factor = np.exp(exponent - kappa * np.exp(exponent))

    dphi_dot_dt = (
        -3.0 * hubble * phi_dot
        + (kappa * V_zero / (b * M_pl)) * exp_factor
        - g**2 * chi_sq_avg * (phi - phi_sbp)
    )

    # ------------------------------------------------------------------
    dchi_dt = chi_dot

    M_chi_sq = (
        (k**2) / (a**2)
        + g**2 * (phi - phi_sbp)**2
        - lam * f**2
        + 3.0 * lam * chi_sq_avg
        + theta_dot_sq_avg
        + (1.0 / a**2) * grad_theta_sq_avg
    )

    dchi_dot_dt = -3.0 * hubble * chi_dot - M_chi_sq * chi

    # ------------------------------------------------------------------
    dtheta_dt = theta_dot

    if chi_sq_avg == 0:
        raise ValueError("chi_sq_avg is zero")

    dtheta_dot_dt = (
        - (2.0 * chi_chidot_avg / chi_sq_avg + 3.0 * hubble) * theta_dot
        - (k**2 / a**2) * theta
    )

    return np.concatenate([
        np.array([dphi_dt, dphi_dot_dt]),
        dchi_dt, dchi_dot_dt,
        dtheta_dt, dtheta_dot_dt
    ])