import numpy as np
import matplotlib.pyplot as plt

# ---- parameters ----
V0 = 1.0
kappa = 1.0
b = 1.0

# ---- phi range ----
phi_min, phi_max = -10.0, 5.0
phi = np.linspace(phi_min, phi_max, 2000)

# ---- potential ----
V = V0 * np.exp(-kappa * np.exp(phi / b))

# ---- plot ----
plt.figure()
plt.plot(phi, V)
plt.xlabel(r"$\phi$")
plt.ylabel(r"$V(\phi)$")
plt.title(r"$V(\phi)=V_0\exp[-\kappa e^{\phi/b}]$")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- optional: plot log(V) ----
plt.figure()
plt.plot(phi, np.log(V))
plt.xlabel(r"$\phi$")
plt.ylabel(r"$\ln V(\phi)$")
plt.title(r"$\ln V(\phi)$")
plt.grid(True)
plt.tight_layout()
plt.show()
