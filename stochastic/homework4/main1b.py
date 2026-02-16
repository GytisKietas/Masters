import numpy as np

N = 40
alpha = 0.5
k0 = 10
max_steps = 20000
size = N + 1
P = np.zeros((size, size))

# State 0 absorbing
P[0, 0] = 1.0

# Interior states 1..38
for i in range(1, N - 1):
    P[i, i - 1] = 0.5
    P[i, i + 1] = 0.5

# State 39 (elastic boundary at 40)
i = N - 1
P[i, i - 1] = 0.5
P[i, i]     = 0.5 * (1 - alpha)
P[i, i + 1] = 0.5 * alpha

# State 40 absorbing
P[N, N] = 1.0

p = np.zeros(size)
p[k0] = 1.0


S = 0.0

for n in range(max_steps):
    S += p[1:N].sum()
    p = p @ P

print("Sum S =", S)
