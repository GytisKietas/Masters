import numpy as np
from scipy.integrate import solve_ivp


class ScipyIVPIntegrator:
    def __init__(self, rhs, method="DOP853", rtol=1e-9, atol=1e-12):
        self.rhs = rhs
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def integrate(self, t0, t_end, y0, dt=None, t_eval=None):
        y0 = np.asarray(y0, dtype=float)

        if t_eval is None and dt is not None:
            n_steps = int(np.floor((t_end - t0) / dt))
            t_eval = t0 + np.arange(n_steps + 1) * dt
            if t_eval[-1] < t_end:
                t_eval = np.append(t_eval, t_end)

        sol = solve_ivp(
            fun=self.rhs,
            t_span=(t0, t_end),
            y0=y0,
            method=self.method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
        )

        if not sol.success:
            raise RuntimeError(f"SciPy solver failed: {sol.message}")

        # transpose to shape (m, n) for easier CSV / comparison
        return sol.t, sol.y.T
