import numpy as np
import matplotlib.pyplot as plt
import math

p = 0.7
N = 10
q = 1 - p
k_max = 30

k_vals = np.arange(0, k_max+1)
pk = [math.comb(k + N - 1, k) * (p**N) * (q**k) for k in k_vals]
pk = np.array(pk)

sum_pk = np.sum(pk)

mu_num = np.sum(k_vals * pk)
var_num = np.sum((k_vals - mu_num)**2 * pk)

mu_analytical = N * (1 - p) / p
var_analytical = N * (1 - p) / (p**2)

print("Sum of probabilities:", sum_pk)
print("Numerical mean:", mu_num)
print("Analytical mean:", mu_analytical)
print("Numerical variance:", var_num)
print("Analytical variance:", var_analytical)

plt.figure(figsize=(8,5))
plt.plot(k_vals, pk, marker='o', linestyle='-', color='b')
plt.title(f"Negative Binomial PMF (p={p}, N={N})")
plt.xlabel("k (number of misses before Nth hit)")
plt.ylabel("P(Y=k)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("exercise2.png", dpi=300, bbox_inches="tight")
plt.close()