#!/usr/bin/env python3
"""
Inflaton evolution in e-fold time N for
V(phi) = V0 * exp[-kappa * exp(phi/b)]
using the N-form equation from the notes (Mpl=1):

phi'' + (3 - phi'^2/2) phi' + (3 - phi'^2/2) (V_phi/V) = 0
epsilon = phi'^2/2, inflation ends at epsilon = 1.

Reference: "Baigiamasis darbas.pdf" equations around the N-form EOM.  (Mpl=1)
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt


def vphi_over_v(phi: float, kappa: float, b: float) -> float:
    """Compute V_{,phi}/V for V = V0 exp[-kappa exp(phi/b)]."""
    # V_phi/V = d/dphi ln V = -(kappa/b) exp(phi/b)
    return -(kappa / b) * np.exp(phi / b)


def rhs(N: float, y: np.ndarray, kappa: float, b: float) -> np.ndarray:
    """
    First-order system in N:
      y = [phi, phip], where phip = dphi/dN
      dy/dN = [phip, phipp]
    """
    phi, phip = float(y[0]), float(y[1])
    wphi = vphi_over_v(phi, kappa=kappa, b=b)

    # From notes (Mpl=1):
    # phi'' + (3 - phip^2/2)*phip + (3 - phip^2/2)*(V_phi/V) = 0
    # => phipp = - (3 - phip^2/2) * (phip + V_phi/V)
    pref = (3.0 - 0.5 * phip * phip)
    phipp = -pref * (phip + wphi)

    return np.array([phip, phipp], dtype=float)


def rk4_step(fun, N: float, y: np.ndarray, h: float, *args) -> np.ndarray:
    """One RK4 step y(N+h) from y(N)."""
    k1 = fun(N, y, *args)
    k2 = fun(N + 0.5 * h, y + 0.5 * h * k1, *args)
    k3 = fun(N + 0.5 * h, y + 0.5 * h * k2, *args)
    k4 = fun(N + h, y + h * k3, *args)
    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_phi(
    phi0: float,
    kappa: float,
    b: float,
    N_max: float = 80.0,
    h: float = 1e-3,
    phip0: float | None = None,
    stop_at_epsilon_1: bool = True,
) -> dict[str, np.ndarray]:
    """
    Integrate until N_max or until epsilon reaches 1.
    """
    # Slow-roll init if not provided:
    # In N-variable slow-roll gives phi' ≈ - V_phi/V  (Mpl=1)
    if phip0 is None:
        phip0 = -vphi_over_v(phi0, kappa=kappa, b=b)

    N_list = [0.0]
    y_list = [np.array([phi0, phip0], dtype=float)]

    eps_list = [0.5 * phip0 * phip0]

    n_steps = int(np.ceil(N_max / h))
    for _ in range(n_steps):
        N = N_list[-1]
        y = y_list[-1]

        y_next = rk4_step(rhs, N, y, h, kappa, b)
        N_next = N + h

        phi_next, phip_next = float(y_next[0]), float(y_next[1])
        eps_next = 0.5 * phip_next * phip_next

        # Stop when epsilon crosses 1 (simple bracket + linear interpolation)
        if stop_at_epsilon_1 and eps_list[-1] < 1.0 <= eps_next:
            # Interpolate to estimate N_end and y_end
            t = (1.0 - eps_list[-1]) / (eps_next - eps_list[-1] + 1e-30)
            N_end = N + t * h
            y_end = y + t * (y_next - y)
            phi_end, phip_end = float(y_end[0]), float(y_end[1])
            eps_end = 0.5 * phip_end * phip_end

            N_list.append(N_end)
            y_list.append(np.array([phi_end, phip_end], dtype=float))
            eps_list.append(eps_end)
            break

        N_list.append(N_next)
        y_list.append(y_next)
        eps_list.append(eps_next)

    arrN = np.array(N_list)
    arrY = np.vstack(y_list)
    arrE = np.array(eps_list)

    return {
        "N": arrN,
        "phi": arrY[:, 0],
        "phip": arrY[:, 1],
        "epsilon": arrE,
    }


def main():
    p = argparse.ArgumentParser(description="Inflaton evolution phi(N) using RK4 (Mpl=1).")
    p.add_argument("--phi0", type=float, default=-2.5, help="Initial phi at N=0.")
    p.add_argument("--kappa", type=float, default=1.0, help="kappa in V(phi)=V0 exp[-kappa exp(phi/b)].")
    p.add_argument("--b", type=float, default=1.0, help="b in V(phi)=V0 exp[-kappa exp(phi/b)].")
    p.add_argument("--Nmax", type=float, default=80.0, help="Max e-folds to integrate if epsilon<1.")
    p.add_argument("--h", type=float, default=1e-3, help="Step size in N.")
    p.add_argument("--phip0", type=float, default=None, help="Initial phi' (overrides slow-roll init).")
    p.add_argument("--no_stop", action="store_true", help="Do not stop at epsilon=1.")
    args = p.parse_args()

    out = simulate_phi(
        phi0=args.phi0,
        kappa=args.kappa,
        b=args.b,
        N_max=args.Nmax,
        h=args.h,
        phip0=args.phip0,
        stop_at_epsilon_1=(not args.no_stop),
    )

    N = out["N"]
    phi = out["phi"]
    eps = out["epsilon"]

    print(f"Integrated to N = {N[-1]:.6f}")
    print(f"Final epsilon = {eps[-1]:.6f}  (inflation ends at epsilon=1)")
    print(f"Final phi = {phi[-1]:.6f}")

    # Plots
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(N, phi)
    ax[0].set_ylabel(r"$\phi(N)$")
    ax[0].grid(True)

    ax[1].plot(N, eps)
    ax[1].axhline(1.0, linestyle="--")
    ax[1].set_xlabel(r"$N=\ln a$")
    ax[1].set_ylabel(r"$\epsilon(N)=\phi'^2/2$")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
