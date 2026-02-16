import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# Observed data
days = np.arange(1, 15)
I_obs = np.array([3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4], dtype=float)
C_obs = np.array([0, 0, 0, 0, 9, 17, 105, 162, 176, 166, 150, 85, 47, 20], dtype=float)
N = 763.0

# Use t=0 for day 1, ..., t=13 for day 14
t_obs = np.arange(0, 14, dtype=float)

# -----------------------------
# SEICR mean-field ODE (for fast fitting + long-run stats)
# infection force uses (E + k I)
# -----------------------------
def seicr_rhs(t, y, beta, k, alpha, gamma, delta):
    S, E, I, C, R = y
    lam = beta * (E + k * I) / N
    dS = -lam * S
    dE = lam * S - alpha * E
    dI = alpha * E - gamma * I
    dC = gamma * I - delta * C
    dR = delta * C
    return [dS, dE, dI, dC, dR]

# -----------------------------
# Fit (deterministic) to get a reasonable parameter set quickly
# We'll FIX k to a plausible "ill are partially infectious" value.
# Then fit beta, alpha, gamma, delta.
# -----------------------------
# k_fixed = 0.001  # ill infectiousness factor (reduced vs presymptomatic)
k_fixed = 0.1

def simulate_ode(beta, alpha, gamma, delta):
    I0 = float(I_obs[0])  # day1 ill
    E0 = 0
    C0 = 0.0
    R0 = 0.0
    S0 = N - I0
    y0 = [S0, E0, I0, C0, R0]
    sol = solve_ivp(
        seicr_rhs, (0, 13), y0,
        t_eval=t_obs,
        args=(beta, k_fixed, alpha, gamma, delta),
        rtol=1e-8, atol=1e-10
    )
    return sol.y

def residuals(theta):
    beta, alpha, gamma, delta = np.exp(theta)
    Y = simulate_ode(beta, alpha, gamma, delta)
    I_hat, C_hat = Y[2], Y[3]

    rI = (I_hat - I_obs) / np.sqrt(I_obs + 1.0)
    rC = (C_hat - C_obs) / np.sqrt(C_obs + 1.0)
    return np.concatenate([rI, rC])

theta0 = np.log([1.0, 1/1.2, 1/3.0, 1/3.0])
res = least_squares(residuals, theta0, max_nfev=8000)
beta, alpha, gamma, delta = np.exp(res.x)

print("\nFITTED PARAMETERS (deterministic fit, used for stochastic sims)")
print(f"beta = {beta:.4f}")
print(f"k = {k_fixed:.2f}")
print(f"alpha = {alpha:.4f}")
print(f"gamma = {gamma:.4f}")
print(f"delta = {delta:.4f}")

print(f"mean incubation = {1/alpha:.3f} days (1/alpha)")
print(f"mean ill-in-bed  = {1/gamma:.3f} days (1/gamma)")
print(f"mean convalescent= {1/delta:.3f} days (1/delta)")


# Gillespie (SSA) simulation
rng = np.random.default_rng(1)

def gillespie_one(T=13.0):
    S = int(round(N - I_obs[0]))
    E = 0
    I = int(round(I_obs[0]))
    C = 0
    R = 0
    cum_ill = I

    out_E = np.zeros_like(t_obs, dtype=float)
    out_I = np.zeros_like(t_obs, dtype=float)
    out_C = np.zeros_like(t_obs, dtype=float)

    t = 0.0
    idx = 0

    # record at t=0
    while idx < len(t_obs) and t_obs[idx] <= t:
        out_E[idx], out_I[idx], out_C[idx] = E, I, C
        idx += 1

    while t < T:
        rate_inf = beta * S * (E + k_fixed * I) / N
        rate_EI  = alpha * E
        rate_IC  = gamma * I
        rate_CR  = delta * C

        rates = np.array([rate_inf, rate_EI, rate_IC, rate_CR], dtype=float)
        a0 = rates.sum()
        if a0 <= 0:
            break
        t += rng.exponential(1.0 / a0)

        while idx < len(t_obs) and t_obs[idx] <= t:
            out_E[idx], out_I[idx], out_C[idx] = E, I, C
            idx += 1

        r = rng.random() * a0
        if r < rates[0]:  # S -> E
            if S > 0:
                S -= 1
                E += 1
        elif r < rates[0] + rates[1]:  # E -> I
            if E > 0:
                E -= 1
                I += 1
                cum_ill += 1
        elif r < rates[0] + rates[1] + rates[2]:  # I -> C
            if I > 0:
                I -= 1
                C += 1
        else:  # C -> R
            if C > 0:
                C -= 1
                R += 1

    while idx < len(t_obs):
        out_E[idx], out_I[idx], out_C[idx] = E, I, C
        idx += 1

    return out_E, out_I, out_C, cum_ill

def summarize(arr):
    mean = arr.mean(axis=0)
    lo   = np.quantile(arr, 0.025, axis=0)
    hi   = np.quantile(arr, 0.975, axis=0)
    return mean, lo, hi

M = 2000
E_all = np.zeros((M, len(t_obs)))
I_all = np.zeros_like(E_all)
C_all = np.zeros_like(E_all)
cum_all = np.zeros(M)

for m in range(M):
    E_all[m], I_all[m], C_all[m], cum_all[m] = gillespie_one()

I_mean, I_lo, I_hi = summarize(I_all)
C_mean, C_lo, C_hi = summarize(C_all)

Y14 = simulate_ode(beta, alpha, gamma, delta)
S14, E14, I14, C14, R14 = Y14

plt.figure()
plt.plot(days, I_obs, marker="o", linestyle="none", label="Observed I")
plt.plot(days, I14, label="Deterministic I")
plt.plot(days, C_obs, marker="s", linestyle="none", label="Observed C")
plt.plot(days, C14, label="Deterministic C")
plt.xlabel("Day of epidemic")
plt.ylabel("Number of pupils")
plt.title("Deterministic mean-field")
plt.legend()
plt.tight_layout()
plt.show()

#-------------------------
t_long = np.linspace(0, 80, 801)
I0 = float(I_obs[0])
y0 = [N - I0, 0.0, I0, 0.0, 0.0]
sol_long = solve_ivp(
    seicr_rhs, (0, 80), y0, t_eval=t_long,
    args=(beta, k_fixed, alpha, gamma, delta),
    rtol=1e-8, atol=1e-10
)
S_l, E_l, I_l, C_l, R_l = sol_long.y

ever_infected_det = N - S_l[-1]
peak_EI_det = np.max(E_l + I_l)
day_peak_det = 1 + t_long[np.argmax(E_l + I_l)]

EI_all = E_all + I_all
peak_EI_stoch = EI_all.max(axis=1)
day_peak_stoch = 1 + t_obs[EI_all.argmax(axis=1)]

print("\nMODEL-BASED QUANTITIES")
print(f"(a) Total ever became ill (stochastic mean) = {cum_all.mean():.1f} pupils (compare: 512)")
print(f"(b) Mean infection->symptoms = 1/alpha = {1/alpha:.3f} days")
print(f"(c) Mean days ill in bed     = 1/gamma = {1/gamma:.3f} days")
print(f"    Mean extra convalescent  = 1/delta = {1/delta:.3f} days")
print(f"(d) Peak of (E+I) deterministic: day {day_peak_det:.2f}, size {peak_EI_det:.1f}")



v_list = np.arange(0.1, 1.0, 0.1)
frac_unvacc_infected = []

for v in v_list:
    V = v * N
    S0v = (1 - v) * N - I0
    if S0v < 0:
        frac_unvacc_infected.append(np.nan)
        continue

    y0v = [S0v, 0.0, I0, 0.0, V]
    solv = solve_ivp(
        seicr_rhs, (0, 80), y0v, t_eval=[80],
        args=(beta, k_fixed, alpha, gamma, delta),
        rtol=1e-8, atol=1e-10
    )
    S_end = solv.y[0, -1]
    infected_unvacc = (S0v - S_end) + I0
    frac_unvacc_infected.append(infected_unvacc / ((1 - v) * N))

plt.figure()
plt.plot(v_list * 100, np.array(frac_unvacc_infected) * 100, marker="o")
plt.xlabel("Vaccinated before return (%)")
plt.ylabel("Infected among unvaccinated (%)")
plt.title("Vaccination impact predicted by the model")
plt.tight_layout()
plt.show()
