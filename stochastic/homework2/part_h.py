import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

P_col = np.array([[0.0, 0.5, 0.0],
                  [1.0, 0.0, 1.0],
                  [0.0, 0.5, 0.0]])

rng = np.random.default_rng(1)

def step(state):
    probs = P_col[:, state]
    return rng.choice(3, p=probs)

def first_return_time_to_1():
    s = 0
    t = 0
    while True:
        s = step(s)
        t += 1
        if s == 0:
            return t

N = 100_000
samples = [first_return_time_to_1() for _ in range(N)]
counts = Counter(samples)

ns = sorted(counts.keys())
pmf_emp = np.array([counts[n]/N for n in ns])

def f11(n):
    if n % 2 == 1:
        return 0.0
    m = n // 2
    if m >= 1:
        return (0.5)**m
    return 0.0

pmf_th = np.array([f11(n) for n in ns])

plt.figure()
plt.bar(ns, pmf_emp, width=0.8, label="empirical")
plt.plot(ns, pmf_th, marker="o", linestyle="-", label="analytical")
plt.xlabel("n (steps to first return to state 1)")
plt.ylabel("probability")
plt.title("Distribution of first return time to state 1")
plt.savefig("part_h.png", dpi=300, bbox_inches="tight")
plt.close()

print("Empirical mean first return time to state 1:", np.mean(samples))
