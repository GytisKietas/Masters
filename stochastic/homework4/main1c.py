import numpy as np

N = 40
alpha = 0.5
k0 = 10

size = N + 1
P = np.zeros((size, size))

P[0, 0] = 1.0

for i in range(1, N - 1):
    P[i, i - 1] = 0.5
    P[i, i + 1] = 0.5

i = N - 1
P[i, i - 1] = 0.5
P[i, i]     = 0.5 * (1 - alpha)
P[i, i + 1] = 0.5 * alpha
P[N, N] = 1.0

T = P[1:N, 1:N]
I = np.eye(T.shape[0])
F = np.linalg.inv(I - T)

tau_10 = F[k0 - 1].sum()

print("Mean duration (tau_10) =", tau_10)
