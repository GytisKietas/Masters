import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# define symbols
delta, epsilon = sp.symbols('delta epsilon')

# build your matrix
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

D = sp.Matrix([
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
detD = D.det()

# turn into numeric function
det_func_A = sp.lambdify((delta, epsilon), detA, 'numpy')
det_func_B = sp.lambdify((delta, epsilon), detB, 'numpy')
det_func_C = sp.lambdify((delta, epsilon), detC, 'numpy')
det_func_D = sp.lambdify((delta, epsilon), detD, 'numpy')

delta_vals = np.linspace(-5, 20, 800)  # x range
eps_vals = np.linspace(0, 60, 800)   # y range
D, E = np.meshgrid(delta_vals, eps_vals)

#evaluate determinant on grid
Z_A = det_func_A(D, E)
Z_B = det_func_B(D, E)
Z_C = det_func_C(D, E)
Z_D = det_func_D(D, E)

# plot contour where det = 0
plt.figure(figsize=(7, 5))
contours_A = plt.contour(D, E, Z_A, levels=[0], colors='blue', linewidths=1.2)
contours_B = plt.contour(D, E, Z_B, levels=[0], colors='blue', linewidths=1.2)
contours_C = plt.contour(D, E, Z_C, levels=[0], colors='blue', linewidths=1.2)
contours_D = plt.contour(D, E, Z_D, levels=[0], colors='blue', linewidths=1.2)

plt.xlabel(r'$\delta$')
plt.ylabel(r'$\epsilon$')
plt.title(r'Determinant Zero Curve: $\det(\epsilon,\delta)=0$')
plt.grid(alpha=0.3)
plt.savefig("matheius_curves.png", dpi=300, bbox_inches='tight')
print("Image saved")