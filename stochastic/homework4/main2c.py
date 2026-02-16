import numpy as np
import matplotlib.pyplot as plt

N = 20
b = 0.028
d = 0.025
t_max = 2000
x0 = 5

P = np.zeros((N+1, N+1))
P[0, 0] = 1.0
for i in range(1, N):
    P[i, i-1] = d * i
    P[i, i]   = 1.0 - i * (b + d)
    P[i, i+1] = b * i
P[N, N-1] = d * N
P[N, N]   = 1.0 - d * N

probs = np.zeros((t_max + 1, N + 1))
probs[0, x0] = 1.0

for t in range(t_max):
    probs[t + 1] = probs[t] @ P

# Survival probability
S = 1.0 - probs[:, 0]

plt.figure()
plt.plot(np.arange(t_max + 1), S)
plt.xlabel('Time step t')
plt.ylabel('Survival probability S(t)')
plt.title('Population survival probability')
plt.grid(True)
plt.tight_layout()
plt.show()

sum_S = S.sum()
print(sum_S)
