import numpy as np

P_col = np.array([[0.0, 0.5, 0.0],
                  [1.0, 0.0, 1.0],
                  [0.0, 0.5, 0.0]])

P = P_col.T
n = P.shape[0]

w, v = np.linalg.eig(P.T)
i_max = np.argmax(np.real(w))
p = np.real(v[:, i_max])
p = p / p.sum()

M = np.zeros((n, n))

for i in range(n):
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    A = np.eye(n-1) - P[np.ix_(mask, mask)]
    b = np.ones(n-1)
    m = np.linalg.solve(A, b)
    M[mask, i] = m
    M[i, i] = 1.0 / p[i]

print("Stationary distribution p:", p)
print("Mean first-passage times matrix M (rows = from j, cols = to i):", M)
