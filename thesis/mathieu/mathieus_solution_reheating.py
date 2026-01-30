import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def solve_mathieu_param(m, g, Phi, k,
                        t_span=(0.0, 60.0), chi0=1.0, chi_dot0=0.0,
                        dt=1e-3, rtol=1e-9, atol=1e-12, method="RK45"):
    """
    Solve  chi'' + m^2 (A_k - 2 q cos(2 m t)) chi = 0
    where  A_k = k^2/m^2 + 2 q,  q = g^2 Phi^2 / (4 m^2).
    """
    q = (g**2 * Phi**2) / (4.0 * m**2)
    A_k = (k**2) / (m**2) + 2.0 * q

    def f(t, y):
        chi, chi_dot = y
        omega2 = m**2 * (A_k - 2.0 * q * np.cos(2.0 * m * t))
        return [chi_dot, -omega2 * chi]

    t0, t1 = t_span
    sol = solve_ivp(f, (t0, t1), [chi0, chi_dot0],
                    method=method, rtol=rtol, atol=atol, dense_output=True)

    t = np.arange(t0, t1 + dt, dt)
    chi = sol.sol(t)[0]
    chi_dot = sol.sol(t)[1]

    omega2_t = m**2 * (A_k - 2.0 * q * np.cos(2.0 * m * t))
    omega_t = np.sqrt(np.maximum(omega2_t, 0.0))

    nk = 0.5 * omega_t * ( (chi_dot**2) / (omega_t**2 + 1e-300) + chi**2 ) - 0.5
    nk = np.maximum(nk, 1e-30) #for finite log

    N = m * t / (2.0 * np.pi)

    return t, chi, chi_dot, q, A_k, nk, N

if __name__ == "__main__":
    #Parameters
    m, g, Phi, k = 1.0, 2.0, 3.0, 0.5

    t, chi, chi_dot, q, A_k, nk, N = solve_mathieu_param(
        m, g, Phi, k,
        t_span=(0, 100),
        chi0=1.0, chi_dot0=0.0
    )

    fig, axes = plt.subplots(2, 1, figsize=(5, 10), sharex=True)

    axes[0].plot(N, chi)
    axes[0].set_ylabel(r"$\chi_k$")
    axes[0].set_title("Narrow resonance (Minkowski)")

    axes[1].plot(N, np.log(nk))
    axes[1].set_xlabel(r"Number of inflaton oscillations  $N=mt/2\pi$")
    axes[1].set_ylabel(r"$\ln n_k$")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        rf"Mathieu form: $A_k=\frac{{k^2}}{{m^2}}+2q$, $q=\frac{{g^2\Phi^2}}{{4m^2}}$  "
        rf"(m={m}, g={g}, $\Phi$={Phi}, k={k};  q={q:.4g}, $A_k$={A_k:.4g})"
    )

    plt.tight_layout()
    plt.savefig("mathieu_param_solution.png", dpi=300, bbox_inches="tight")
    print("Saved: mathieu_param_solution.png")
