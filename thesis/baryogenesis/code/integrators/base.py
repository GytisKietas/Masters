from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseIntegrator(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(self, state, t_current, timestep, params):
        """Integration step"""

    def integrate(self, simulation_time, timestep, initial_state, params):
        N_steps = int(simulation_time/ abs(timestep)) + 1
        
        state = np.array(initial_state, dtype=float)
        dim = state.size
        t_current = 0.0

        t_values = np.empty(N_steps)
        states = np.empty((N_steps, dim))

        for i in range(N_steps):
            t_values[i] = t_current
            states[i, :] = state

            logging.info(f"STEP {i} | t: {t_values[i]:.4f}  |  state: {state}")

            state = self.step(state, t_current, timestep, params)

            t_current += timestep
        
        return t_values, states
            