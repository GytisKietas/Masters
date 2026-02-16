# payoff_surface_n3.py
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)

# ----- Payoff table R you provided (rows: A, B, C ; columns: outcomes 000..111) -----
R = np.array([
    [6, 3, 3, 0, 9, 5, 5, 1],  # player A's payoffs r_k
    [6, 3, 9, 5, 3, 0, 5, 1],  # player B
    [6, 9, 3, 5, 3, 5, 0, 1],  # player C
], dtype=float)

# Choose whose payoff to plot on z-axis (0=A, 1=B, 2=C)
who = 0  # A by default

# Fixed choice for player 3: theta3 = pi (always outcome '1' with prob 1)
theta3 = np.pi

# We keep phases at 0 so that the surface aligns with classical C(=0) to D(=pi) on each axis.
phi1 = phi2 = 0.0

# Grid of thetas for players 1 and 2
n_pts = 121
theta1_vals = np.linspace(0, np.pi, n_pts)  # x-axis
theta2_vals = np.linspace(0, np.pi, n_pts)  # y-axis

# Helper: outcome probability under independent measurements given thetas
def outcome_prob(theta_list, bitstring):
    # p(0)=cos^2(theta/2), p(1)=sin^2(theta/2); phases drop out under Z-basis measurement
    p = 1.0
    for th, b in zip(theta_list, bitstring):
        if b == 0:
            p *= np.cos(th/2.0)**2
        else:
            p *= np.sin(th/2.0)**2
    return p

# Precompute bitstrings and map to column index 0..7 in binary order (000..111)
bitstrings = list(product([0, 1], repeat=3))
# columns follow the standard binary order
col_index = {bs: int("".join(map(str, bs)), 2) for bs in bitstrings}

# Build the surface: expected payoff for 'who' vs (theta1, theta2) with theta3 fixed to pi
Z = np.zeros((n_pts, n_pts), dtype=float)

for i, th1 in enumerate(theta1_vals):
    for j, th2 in enumerate(theta2_vals):
        # thetas for players 1,2,3
        thetas = (th1, th2, theta3)
        # expected value: sum_k r_k * P(k)
        EV = 0.0
        for bs in bitstrings:
            k = col_index[bs]
            pk = outcome_prob(thetas, bs)
            EV += R[who, k] * pk
        Z[j, i] = EV  # note: j,i so y varies along rows, x along cols

# ----- Plot -----
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(theta1_vals, theta2_vals)
surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, rstride=1, cstride=1, alpha=0.95)

ax.set_xlabel(r'Player 1 angle $\theta_1$  (C $\to$ D)')
ax.set_ylabel(r'Player 2 angle $\theta_2$  (C $\to$ D)')
ax.set_zlabel(r'Expected payoff $E_{\mathrm{A}}$')

# tick labels to echo C (0) and D (pi)
ax.set_xticks([0, np.pi/2, np.pi])
ax.set_xticklabels([r'$C$', r'$\pi/2$', r'$D$'])
ax.set_yticks([0, np.pi/2, np.pi])
ax.set_yticklabels([r'$C$', r'$\pi/2$', r'$D$'])

# Mention the fixed choice for player 3
ax.text2D(0.02, 0.95, r'Fixed: $\theta_3=\pi$ (Player 3 $\equiv D$)', transform=ax.transAxes)

fig.tight_layout()
# plt.show()
# Optionally save:
fig.savefig("payoff_surface_n3_A_theta3_pi.png", dpi=200, bbox_inches="tight")
