import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Parameters
N = 20
b = 0.028
d = 0.025
t_max = 2000
x0 = 5 

P = np.zeros((N + 1, N + 1))

# State 0 absorbing
P[0, 0] = 1.0

# States 1..N-1
for i in range(1, N):
    P[i, i - 1] = d * i
    P[i, i] = 1.0 - i * (b + d)
    P[i, i + 1] = b * i

# State N (no births)
P[N, N - 1] = d * N
P[N, N]     = 1.0 - d * N

probs = np.zeros((t_max + 1, N + 1))
probs[0, x0] = 1.0

for t in range(t_max):
    probs[t + 1] = probs[t] @ P


states = np.arange(N + 1)
X_mean = probs @ states


def biexp(t, A, t1, t2):
    return A * np.exp(-t / t1) + (x0 - A) * np.exp(-t / t2)

t_vals = np.arange(t_max + 1, dtype=float)

p0 = [2.0, 300.0, 3000.0]


popt, pcov = curve_fit(biexp, t_vals, X_mean, p0=p0, maxfev=200_000)
A_fit, t1_fit, t2_fit = popt

print("Fitted parameters:")
print(f"A  = {A_fit:.3f}")
print(f"t1 = {t1_fit:.1f}")
print(f"t2 = {t2_fit:.1f}")


plt.figure()
plt.plot(t_vals, X_mean, label=r'$\langle X(t)\rangle$ (simulation)')
plt.plot(t_vals, biexp(t_vals, *popt), '--',
         label='Bi-exponential fit')

plt.xlabel('Time step t')
plt.ylabel(r'Average population $\langle X(t)\rangle$')
plt.title('Average population vs time with bi-exponential fit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
