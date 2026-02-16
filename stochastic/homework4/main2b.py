import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
N = 20
b = 0.028
d = 0.025
t_max = 2000
x0 = 5

P = np.zeros((N+1, N+1))

# State 0: absorbing
P[0, 0] = 1.0

# States 1..N-1
for i in range(1, N):
    P[i, i-1] = d * i
    P[i, i]   = 1.0 - i * (b + d)
    P[i, i+1] = b * i

# State N: no births
P[N, N-1] = d * N
P[N, N]   = 1.0 - d * N


probs = np.zeros((t_max + 1, N + 1))
probs[0, x0] = 1.0

for t in range(t_max):
    probs[t + 1] = probs[t] @ P

T = np.arange(t_max + 1)
I = np.arange(N + 1)

T_grid, I_grid = np.meshgrid(T, I)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(T_grid, I_grid, probs.T)

ax.set_xlabel('Time t')
ax.set_ylabel('Population size i')
ax.set_zlabel('Probability p_i(t)')
ax.set_title('Evolution of population size probabilities')

plt.tight_layout()
plt.show()
