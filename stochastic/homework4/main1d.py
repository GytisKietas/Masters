import numpy as np
import matplotlib.pyplot as plt

N = 40
alpha = 0.5
size = N + 1

P = np.zeros((size, size))

# State 0: absorbing
P[0, 0] = 1.0

# States 1..38: simple symmetric walk
for i in range(1, N - 1):
    P[i, i - 1] = 0.5
    P[i, i + 1] = 0.5

# State 39: elastic/absorbing boundary at 40
i = N - 1
P[i, i - 1] = 0.5
P[i, i]     = 0.5 * (1 - alpha)
P[i, i + 1] = 0.5 * alpha

# State 40: absorbing
P[N, N] = 1.0

T = P[1:N, 1:N]

I = np.eye(T.shape[0])
F = np.linalg.inv(I - T)

taus = F.sum(axis=1)
k_vals = np.arange(1, N)

print("tau_10 =", taus[10 - 1])

plt.figure()
plt.plot(k_vals, taus, 'o-')
plt.xlabel('Initial position k')
plt.ylabel(r'Mean duration $\tau_k$')
plt.title(r'Mean duration of the walk vs starting position $k$')
plt.grid(True)
plt.tight_layout()
plt.show()
