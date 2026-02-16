import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# -------------------------
# Data (Table 4): days 1..14
# -------------------------
days = np.arange(1, 15)
t_obs = np.arange(0, 14, dtype=float)   # t=0 corresponds to day 1
I_obs = np.array([3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4], dtype=float)
C_obs = np.array([0, 0, 0, 0, 9, 17, 105, 162, 176, 166, 150, 85, 47, 20], dtype=float)
N = 763.0

# -------------------------
# SEICR deterministic ODE model
# -------------------------
def seicr_rhs(t, y, beta, k, alpha, gamma, delta):
    S, E, I, C, R = y
    lam = beta * (E + k*I) / N
    dS = -lam * S
    dE = lam * S - alpha * E
    dI = alpha * E - gamma * I
    dC = gamma * I - delta * C
    dR = delta * C
    return [dS, dE, dI, dC, dR]

def simulate(beta, k, alpha, gamma, delta, E0):
    I0 = float(I_obs[0])
    C0 = 0.0
    R0 = 0.0
    S0 = N - E0 - I0
    if S0 < 0:
        return None

    y0 = [S0, E0, I0, C0, R0]
    sol = solve_ivp(
        seicr_rhs, (0, 13), y0, t_eval=t_obs,
        args=(beta, k, alpha, gamma, delta),
        rtol=1e-7, atol=1e-9
    )
    if not sol.success:
        return None
    return sol.y  # shape (5, len(t_obs))

# -------------------------
# Fit parameters
# - rates are positive -> optimize logs
# - k in (0,1) -> logistic transform
# - E0 in [0,200] -> logistic transform with cap
# -------------------------
def unpack(theta):
    log_beta, log_alpha, log_gamma, log_delta, logit_k, logit_E0 = theta

    beta  = np.exp(log_beta)
    alpha = np.exp(log_alpha)
    gamma = np.exp(log_gamma)
    delta = np.exp(log_delta)

    k  = 1.0 / (1.0 + np.exp(-logit_k))          # (0,1)
    E0 = 200.0 * (1.0 / (1.0 + np.exp(-logit_E0))) # [0,200]

    return beta, k, alpha, gamma, delta, E0

def residuals(theta):
    beta, k, alpha, gamma, delta, E0 = unpack(theta)
    Y = simulate(beta, k, alpha, gamma, delta, E0)
    if Y is None:
        return np.ones(28) * 1e6

    I_hat = Y[2]
    C_hat = Y[3]

    # weighted residuals so huge counts don't dominate too much
    rI = (I_hat - I_obs) / np.sqrt(I_obs + 1.0)
    rC = (C_hat - C_obs) / np.sqrt(C_obs + 1.0)
    return np.concatenate([rI, rC])

# Initial guess
theta0 = np.array([
    np.log(2.0),    # beta
    np.log(1.0),    # alpha
    np.log(1/2.0),  # gamma
    np.log(1/2.0),  # delta
    -2.0,           # k ~ 0.12
    -2.0            # E0 ~ 24 out of 200
], dtype=float)

res = least_squares(residuals, theta0, max_nfev=1200)
beta, k, alpha, gamma, delta, E0 = unpack(res.x)

print("FITTED PARAMETERS (ODE fit)")
print(f"beta   = {beta:.4f} per day")
print(f"k      = {k:.4f} (ill infectiousness factor)")
print(f"alpha  = {alpha:.4f} per day  -> mean incubation = {1/alpha:.3f} days")
print(f"gamma  = {gamma:.4f} per day  -> mean ill-in-bed  = {1/gamma:.3f} days")
print(f"delta  = {delta:.4f} per day  -> mean convalescent= {1/delta:.3f} days")
print(f"E0     = {E0:.2f} pupils (latent at day 1)")
print(f"least_squares cost = {res.cost:.3f}")

# -------------------------
# Plot fitted curves
# -------------------------
Y = simulate(beta, k, alpha, gamma, delta, E0)
S_hat, E_hat, I_hat, C_hat, R_hat = Y

plt.figure()
plt.plot(days, I_obs, "o", label="Observed ill (I)")
plt.plot(days, C_obs, "s", label="Observed recovering (C)")
plt.plot(days, I_hat, "-", label="Model I(t)")
plt.plot(days, C_hat, "-", label="Model C(t)")
plt.xlabel("Day of epidemic")
plt.ylabel("Number of pupils")
plt.title("SEICR deterministic ODE fit (I and C)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(days, E_hat, "-", label="Model E(t) (latent)")
plt.xlabel("Day of epidemic")
plt.ylabel("Number of pupils")
plt.title("Latent compartment E(t) from fitted model")
plt.legend()
plt.tight_layout()
plt.show()
