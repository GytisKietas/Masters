import numpy as np

import matplotlib
matplotlib.use("TkAgg")   # must be before importing pyplot

import matplotlib.pyplot as plt

p = 0.7
q = 1 - p
k_max = 20

k_vals = np.arange(0, k_max+1)
pk = (q**k_vals) * p

sum_pk = np.sum(pk)

mu_num = np.sum(k_vals * pk)
var_num = np.sum((k_vals - mu_num)**2 * pk)

mu_analytical = (1-p)/p
var_analytical = (1-p)/(p**2)

print("Sum of probabilities:", sum_pk)
print("Numerical mean:", mu_num)
print("Analytical mean:", mu_analytical)
print("Numerical variance:", var_num)
print("Analytical variance:", var_analytical)

plt.figure(figsize=(8,5))
plt.plot(k_vals, pk, marker='o', linestyle='-', color='b')
plt.title("Geometric distribution PMF (p=0.7)")
plt.xlabel("k (number of misses before first hit)")
plt.ylabel("P(X=k)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
# plt.savefig("exercise1.png", dpi=300, bbox_inches="tight")
# plt.close()