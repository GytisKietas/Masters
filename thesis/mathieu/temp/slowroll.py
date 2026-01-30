import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters (edit these)
# ----------------------------
Mpl   = 1.0      # set M_pl = 1 if you're using reduced Planck units
kappa = 0.5
b     = 2.0

# Initial condition: phi(N0) = phi0
N0   = 0.0
phi0 = -3.0 * Mpl

# Choose N range (the code will automatically stop before the log blows up)
N_min = N0
N_max_try = 200.0  # "attempt" max; actual max may be smaller due to domain

# ----------------------------
# Solution phi(N)
# phi(N) = - b Mpl * ln[ exp(-phi0/(b Mpl)) - (kappa/b^2)(N-N0) ]
# ----------------------------
A = np.exp(-phi0 / (b * Mpl))

# Domain limit (where argument -> 0)
N_end = N0 + (b**2 / kappa) * A

# Pick a safe max N (leave a little margin)
N_max = min(N_max_try, N_end * 0.999)

if N_max <= N_min:
    raise ValueError(
        "Your parameters/initial condition give no valid N-range. "
        "Try changing phi0, kappa, or b."
    )

N = np.linspace(N_min, N_max, 2000)
arg = A - (kappa / b**2) * (N - N0)
phi = -b * Mpl * np.log(arg)

# ----------------------------
# Plot
# ----------------------------
plt.figure()
plt.plot(N, phi)
plt.xlabel("N  (e-folds)")
plt.ylabel(r"$\phi(N)$")
plt.title(r"$\phi(N)=-bM_{\rm Pl}\ln\!\left[e^{-\phi_0/(bM_{\rm Pl})}-\frac{\kappa}{b^2}(N-N_0)\right]$")
plt.grid(True)
plt.tight_layout()

plt.savefig("slowroll.png", dpi=300, bbox_inches='tight')

plt.show()
