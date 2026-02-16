import numpy as np
import matplotlib.pyplot as plt

N = 40
alpha = 0.5
p = q = 0.5
size = N + 1

P = np.zeros((size, size))

# State 0: absorbing
P[0, 0] = 1.0

# Interior states 1..38
for i in range(1, N - 1):
    P[i, i - 1] = 0.5
    P[i, i + 1] = 0.5

# State 39: elastic boundary at 40
i = N - 1
P[i, i - 1] = 0.5
P[i, i]     = 0.5 * (1 - alpha)
P[i, i + 1] = 0.5 * alpha

# State 40: absorbing
P[N, N] = 1.0

k0 = 10
p_vec = np.zeros(size)
p_vec[k0] = 1.0

max_steps = 5000

p0_hist = []
p40_hist = []
pmid_hist = []

for n in range(max_steps + 1):
    p0_hist.append(p_vec[0])
    p40_hist.append(p_vec[N])
    pmid_hist.append(p_vec[1:N].sum())
    p_vec = p_vec @ P

plt.figure(figsize=(10,5))
plt.plot(p0_hist, label=r'$p_0(n)$ (absorb left)')
plt.plot(p40_hist, label=r'$p_{40}(n)$ (absorb right)')
plt.plot(pmid_hist, label=r'$p_{\rm mid}(n)$ (interior)')

plt.xlabel('n (time steps)')
plt.ylabel('Probability')
plt.title('Evolution of probabilities for the random walk')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()