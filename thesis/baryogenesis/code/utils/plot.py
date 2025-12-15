import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)


def plot(t_values, states, params):
    # assuming layout: [phi, phidot, chi0, chidot0, chi1, chidot1, ...]
    phi     = states[:, 0]
    phidot  = states[:, 1]

    num_modes = params["num_modes"]
    chi = np.zeros((len(t_values), num_modes))

    for m in range(num_modes):
        chi[:, m] = states[:, 2 + 2*m]        # pick only χ_k, skip χ̇_k

    # Plot φ(t)
    plt.figure()
    plt.plot(t_values, phi, label=r'$\phi(t)$')
    plt.xlabel(r'$t$')
    plt.ylabel(r'field value')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot each χ_k(t)
    plt.figure()
    for m in range(num_modes):
        plt.plot(t_values, chi[:, m], label=fr'$\chi_{{k_{m}}}(t)$')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\chi_k$')
    plt.legend()
    plt.tight_layout()
    plt.show()
