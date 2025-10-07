import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def mathieu_solve_scipy(delta, epsilon, t_span=(0.0, 60.0), x0=1.0, v0=0.0,
                        dt=0.001, rtol=1e-9, atol=1e-12, method="RK45"):
    def f(t, y):
        x, v = y
        return [v, -(delta + epsilon*np.cos(2.0*t))*x]

    t0, t1 = t_span
    sol = solve_ivp(f, (t0, t1), [x0, v0], method=method, rtol=rtol, atol=atol, dense_output=True)  
    t = np.arange(t0, t1 + dt, dt)
    x = sol.sol(t)[0]
    v = sol.sol(t)[1]
    return t, x, v


if __name__ == "__main__":
    delta, epsilon = 2.0, 3.2
    t, x, v = mathieu_solve_scipy(delta, epsilon, t_span=(0, 100), x0=1.0, v0=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Normal plot
    axes[0].plot(t, x, label="x(t)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    axes[0].set_title("Linear scale")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Logarithmic y-axis
    axes[1].plot(t, np.abs(x), label="|x(t)|")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x (log scale)")
    axes[1].set_title("Logarithmic scale (y)")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend()

    fig.suptitle(fr"Mathieu equation: $\delta={delta}$, $\varepsilon={epsilon}$")
    plt.tight_layout()

    plt.savefig(f"mathieu_solution_d{delta}_e{epsilon}.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Image saved")
