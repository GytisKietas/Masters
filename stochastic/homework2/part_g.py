import numpy as np
import matplotlib.pyplot as plt

P_col = np.array([[0.0, 0.5, 0.0],
                  [1.0, 0.0, 1.0],
                  [0.0, 0.5, 0.0]])

rng = np.random.default_rng(0)

def step(state):
    probs = P_col[:, state]
    return rng.choice(3, p=probs)

#short trajectory
T_short = 50
x = np.zeros(T_short+1, dtype=int)
x[0] = 0
for t in range(T_short):
    x[t+1] = step(x[t])

plt.figure()
plt.plot(np.arange(T_short+1), x+1, marker="o")
plt.xlabel("step")
plt.ylabel("state")
plt.title("50-step trajectory")
plt.savefig("part_g_1.png", dpi=300, bbox_inches="tight")
plt.close()

#long run
T_long = 200000
state = 0
counts = np.zeros(3, dtype=int)
for _ in range(T_long):
    state = step(state)
    counts[state] += 1
freq = counts / counts.sum()

print("Empirical visit frequencies:", freq)


p = np.array([0.25, 0.50, 0.25])
plt.figure()
idx = np.arange(1,4)
plt.bar(idx-0.15, freq, width=0.3, label="empirical")
plt.bar(idx+0.15, p,   width=0.3, label="analytical")
plt.xticks(idx, ["1","2","3"])
plt.ylabel("probability")
plt.title("Visit frequencies vs stationary distribution")
plt.legend()
plt.savefig("part_g_2.png", dpi=300, bbox_inches="tight")
plt.close()
