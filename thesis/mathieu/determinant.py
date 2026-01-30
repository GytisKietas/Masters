# import sympy as sp
# import numpy as np
# import matplotlib.pyplot as plt

# # define symbols
# delta, epsilon = sp.symbols('delta epsilon')

# # build your matrix
# A = sp.Matrix([
#     [delta, epsilon / 2, 0, 0, 0, 0, 0],
#     [epsilon, delta - 4, epsilon / 2, 0, 0, 0, 0],
#     [0, epsilon / 2, delta - 16, epsilon / 2, 0, 0, 0],
#     [0, 0, epsilon / 2, delta - 36, epsilon / 2,0, 0],
#     [0, 0, 0, epsilon / 2, delta - 64, epsilon / 2, 0],
#     [0, 0, 0, 0, epsilon / 2, delta - 100, epsilon / 2],
#     [0, 0, 0, 0, 0, epsilon / 2, delta - 144]
# ])

# B = sp.Matrix([
#     [delta - 4, epsilon / 2, 0, 0, 0, 0, 0],
#     [epsilon / 2, delta - 16, epsilon / 2, 0, 0, 0, 0],
#     [0, epsilon / 2, delta - 36, epsilon / 2, 0, 0, 0],
#     [0, 0, epsilon / 2, delta - 64, epsilon / 2, 0, 0],
#     [0, 0, 0, epsilon / 2, delta - 100, epsilon / 2, 0],
#     [0, 0, 0, 0, epsilon / 2, delta - 144, epsilon / 2],
#     [0, 0, 0, 0, 0, epsilon / 2, delta - 196]
# ])

# C = sp.Matrix([
#     [delta - 1 + epsilon / 2, epsilon / 2, 0, 0, 0, 0,0],
#     [epsilon / 2, delta - 9, epsilon / 2, 0, 0, 0,0],
#     [0, epsilon / 2, delta - 25, epsilon / 2,0, 0,0],
#     [0, 0, epsilon / 2, delta - 49, epsilon / 2,0,0],
#     [0, 0, 0, epsilon / 2, delta - 81, epsilon / 2,0],
#     [0, 0, 0, 0, epsilon / 2, delta - 121, epsilon / 2],
#     [0, 0, 0, 0, 0, epsilon / 2, delta - 169]
# ])

# D = sp.Matrix([
#     [delta - 1 - epsilon / 2, epsilon / 2, 0, 0, 0, 0,0],
#     [epsilon / 2, delta - 9, epsilon / 2, 0, 0, 0,0],
#     [0, epsilon / 2, delta - 25, epsilon / 2,0, 0,0],
#     [0, 0, epsilon / 2, delta - 49, epsilon / 2,0,0],
#     [0, 0, 0, epsilon / 2, delta - 81, epsilon / 2,0],
#     [0, 0, 0, 0, epsilon / 2, delta - 121, epsilon / 2],
#     [0, 0, 0, 0, 0, epsilon / 2, delta - 169]
# ])

# detA = A.det()
# detB = B.det()
# detC = C.det()
# detD = D.det()

# # turn into numeric function
# det_func_A = sp.lambdify((delta, epsilon), detA, 'numpy')
# det_func_B = sp.lambdify((delta, epsilon), detB, 'numpy')
# det_func_C = sp.lambdify((delta, epsilon), detC, 'numpy')
# det_func_D = sp.lambdify((delta, epsilon), detD, 'numpy')

# delta_vals = np.linspace(-5, 20, 800)  # x range
# eps_vals = np.linspace(0, 60, 800)   # y range
# D, E = np.meshgrid(delta_vals, eps_vals)

# #evaluate determinant on grid
# Z_A = det_func_A(D, E)
# Z_B = det_func_B(D, E)
# Z_C = det_func_C(D, E)
# Z_D = det_func_D(D, E)

# # plot contour where det = 0
# plt.figure(figsize=(7, 5))
# contours_A = plt.contour(D, E, Z_A, levels=[0], colors='blue', linewidths=1.2)
# contours_B = plt.contour(D, E, Z_B, levels=[0], colors='blue', linewidths=1.2)
# contours_C = plt.contour(D, E, Z_C, levels=[0], colors='blue', linewidths=1.2)
# contours_D = plt.contour(D, E, Z_D, levels=[0], colors='blue', linewidths=1.2)

# plt.xlabel(r'$A_k$')
# plt.ylabel(r'$q$')
# plt.title(r'Stability bands')
# plt.grid(alpha=0.3)
# plt.savefig("matheius_curves.png", dpi=300, bbox_inches='tight')
# print("Image saved")
# #THIS IS A TEST

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# define symbols
delta, epsilon = sp.symbols('delta epsilon')

# build matrices (same as yours)
A = sp.Matrix([
    [delta, epsilon / 2, 0, 0, 0, 0, 0],
    [epsilon, delta - 4, epsilon / 2, 0, 0, 0, 0],
    [0, epsilon / 2, delta - 16, epsilon / 2, 0, 0, 0],
    [0, 0, epsilon / 2, delta - 36, epsilon / 2,0, 0],
    [0, 0, 0, epsilon / 2, delta - 64, epsilon / 2, 0],
    [0, 0, 0, 0, epsilon / 2, delta - 100, epsilon / 2],
    [0, 0, 0, 0, 0, epsilon / 2, delta - 144]
])

B = sp.Matrix([
    [delta - 4, epsilon / 2, 0, 0, 0, 0, 0],
    [epsilon / 2, delta - 16, epsilon / 2, 0, 0, 0, 0],
    [0, epsilon / 2, delta - 36, epsilon / 2, 0, 0, 0],
    [0, 0, epsilon / 2, delta - 64, epsilon / 2, 0, 0],
    [0, 0, 0, epsilon / 2, delta - 100, epsilon / 2, 0],
    [0, 0, 0, 0, epsilon / 2, delta - 144, epsilon / 2],
    [0, 0, 0, 0, 0, epsilon / 2, delta - 196]
])

C = sp.Matrix([
    [delta - 1 + epsilon / 2, epsilon / 2, 0, 0, 0, 0,0],
    [epsilon / 2, delta - 9, epsilon / 2, 0, 0, 0,0],
    [0, epsilon / 2, delta - 25, epsilon / 2,0, 0,0],
    [0, 0, epsilon / 2, delta - 49, epsilon / 2,0,0],
    [0, 0, 0, epsilon / 2, delta - 81, epsilon / 2,0],
    [0, 0, 0, 0, epsilon / 2, delta - 121, epsilon / 2],
    [0, 0, 0, 0, 0, epsilon / 2, delta - 169]
])

Dmat = sp.Matrix([
    [delta - 1 - epsilon / 2, epsilon / 2, 0, 0, 0, 0,0],
    [epsilon / 2, delta - 9, epsilon / 2, 0, 0, 0,0],
    [0, epsilon / 2, delta - 25, epsilon / 2,0, 0,0],
    [0, 0, epsilon / 2, delta - 49, epsilon / 2,0,0],
    [0, 0, 0, epsilon / 2, delta - 81, epsilon / 2,0],
    [0, 0, 0, 0, epsilon / 2, delta - 121, epsilon / 2],
    [0, 0, 0, 0, 0, epsilon / 2, delta - 169]
])

detA = A.det()
detB = B.det()
detC = C.det()
detD = Dmat.det()

det_func_A = sp.lambdify((delta, epsilon), detA, 'numpy')
det_func_B = sp.lambdify((delta, epsilon), detB, 'numpy')
det_func_C = sp.lambdify((delta, epsilon), detC, 'numpy')
det_func_D = sp.lambdify((delta, epsilon), detD, 'numpy')

# grid
delta_vals = np.linspace(-5, 20, 800)
eps_vals   = np.linspace(0, 60, 800)
Dgrid, Egrid = np.meshgrid(delta_vals, eps_vals)

# evaluate
Z_A = det_func_A(Dgrid, Egrid)
Z_B = det_func_B(Dgrid, Egrid)
Z_C = det_func_C(Dgrid, Egrid)
Z_D = det_func_D(Dgrid, Egrid)

from scipy.ndimage import label

# ... after you compute `unstable = mask_even | mask_odd` ...
mask_even = (Z_A * Z_B) < 0
mask_odd  = (Z_C * Z_D) < 0
unstable  = mask_even | mask_odd

# Label connected regions in the candidate unstable mask
labels, nlab = label(unstable)   # 8-connectivity by default for boolean grids

# Pick one small-q seed point inside each tongue you want filled
# (A_k ~ n^2 at q~0 for Mathieu tongues)
seed_points = [
    (0.5, 0.5),   # near A_k ~ 0 tongue (if you want it)
    (1.0, 0.5),   # near A_k ~ 1 tongue
    (4.0, 0.5),   # near A_k ~ 4 tongue
    (9.0, 0.5),   # near A_k ~ 9 tongue
    (16.0, 0.5),  # near A_k ~ 16 tongue
]

# Convert seed (A_k, q) to grid indices and collect which components to keep
keep = set()
for Ak0, q0 in seed_points:
    j = np.argmin(np.abs(delta_vals - Ak0))  # x index
    i = np.argmin(np.abs(eps_vals   - q0))   # y index
    lab = labels[i, j]
    if lab != 0:  # 0 means background (not in unstable mask)
        keep.add(lab)

# Build final mask: only those labeled components
tongues = np.isin(labels, list(keep))

plt.figure(figsize=(7, 5))

# Fill ONLY the selected tongue regions
plt.contourf(Dgrid, Egrid, tongues.astype(int), levels=[0.5, 1.5],
             colors=['#4c78a8'], alpha=0.25)

# Draw boundaries on top (your original lines)
plt.contour(Dgrid, Egrid, Z_A, levels=[0], colors='blue', linewidths=1.2)
plt.contour(Dgrid, Egrid, Z_B, levels=[0], colors='blue', linewidths=1.2)
plt.contour(Dgrid, Egrid, Z_C, levels=[0], colors='blue', linewidths=1.2)
plt.contour(Dgrid, Egrid, Z_D, levels=[0], colors='blue', linewidths=1.2)

plt.xlabel(r'$A_k$')
plt.ylabel(r'$q$')
plt.title('Stability bands')
plt.grid(alpha=0.3)
plt.savefig("matheius_curves_filled.png", dpi=300, bbox_inches='tight')
plt.show()



# plt.figure(figsize=(7, 5))

# # fill unstable regions
# plt.contourf(Dgrid, Egrid, unstable.astype(int),
#              levels=[0.5, 1.5], alpha=0.25)

# # draw boundaries on top
# plt.contour(Dgrid, Egrid, Z_A, levels=[0], colors='blue', linewidths=1.2)
# plt.contour(Dgrid, Egrid, Z_B, levels=[0], colors='blue', linewidths=1.2)
# plt.contour(Dgrid, Egrid, Z_C, levels=[0], colors='blue', linewidths=1.2)
# plt.contour(Dgrid, Egrid, Z_D, levels=[0], colors='blue', linewidths=1.2)

# plt.xlabel(r'$A_k$')
# plt.ylabel(r'$q$')
# plt.title('Stability bands')
# plt.grid(alpha=0.3)

# plt.savefig("matheius_curves_filled.png", dpi=300, bbox_inches='tight')
# plt.show()
