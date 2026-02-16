import numpy as np

N=20
b=0.028
d=0.025

Q = np.zeros((N, N))

for i in range(1, N):
    idx = i - 1
    Q[idx, idx] = 1.0 - (b + d) * i
    Q[idx, idx + 1] = b * i

    if i > 1:
        Q[idx, idx - 1] = d * i

Q[N - 1, N - 1] = 1.0 - d * N
Q[N - 1, N - 2] = d * N

I = np.eye(N)
A = I - Q
ones = np.ones(N)

T = np.linalg.solve(A, ones)

T5 = T[4]
print(f"T_5 (expected time to extinction from 5): {T5:.6f}")
