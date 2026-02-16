import numpy as np
import matplotlib.pyplot as plt

P = np.array([
    [0.5, 0, 0],
    [0.5, 0, 1/3],
    [0, 1, 2/3]
])

v0 = np.array([1.0, 0.0, 0.0])

steps = 20

probs = [v0]
for _ in range(steps):
    probs.append(P @ probs[-1])

probs = np.stack(probs, axis=0)

plt.figure(figsize=(8,5))
plt.plot(probs[:, 0], marker='o', linestyle='-', label="State 1")
plt.plot(probs[:, 1], marker='s', linestyle='-', label="State 2")
plt.plot(probs[:, 2], marker='^', linestyle='-', label="State 3")
plt.title("Temporal evolution of state probabilities (column-vector update)")
plt.xlabel("Step")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("exercise3.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved plot to markov_chain_evolution_col.png")
print("Final probabilities:", probs[-1])