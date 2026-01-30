import numpy as np

def harmonic_rhs(t_current, state, params):
    x, v = state

    omega = params.get("omega", 1.0)

    dxdt = v
    dvdt = -omega**2 * x

    return np.array([dxdt, dvdt])


def coupled_oscillators_rhs(t, state, params):
    x1, x2, v1, v2 = state

    omega1 = params.get("omega1", 1.0)
    omega2 = params.get("omega2", 1.0)
    k = params.get("k", 0.1)

    # Equations of motion
    dx1dt = v1
    dx2dt = v2
    dv1dt = -omega1**2 * x1 - k * (x1 - x2)
    dv2dt = -omega2**2 * x2 - k * (x2 - x1)

    return np.array([dx1dt, dx2dt, dv1dt, dv2dt])