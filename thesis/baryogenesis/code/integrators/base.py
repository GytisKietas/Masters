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

    def integrate(self, tspan, initial_state, params, on_save = None):
        t_end = float(tspan["t_end"])
        t0 = float(tspan["t0"])

        simulation_time = np.abs(t_end - t0)
        timestep = float(tspan["dt"])

        N_steps = int(simulation_time/ abs(timestep)) + 1
        
        state = np.array(initial_state, dtype=float)
        dim = state.size
        t_current = t0

        t_values = np.empty(N_steps)
        states = np.empty((N_steps, dim))

        for i in range(N_steps):
            t_values[i] = t_current
            states[i, :] = state

            logging.info(f"STEP {i} | t: {t_values[i]:.4f}  |  state: {state}")

            state = self.step(state, t_current, timestep, params)

            t_current += timestep

            if i % 10000 == 0 and on_save is not None:
                on_save(i, t_values.copy(), states.copy())
        
        return t_values, states
    
    def integrate_conditional(self, tspan, initial_state, params, on_end_condition = None, on_save = None):
        t_end = float(tspan["t_end"])
        t0 = float(tspan["t0"])

        simulation_time = np.abs(t_end - t0)
        timestep = float(tspan["dt"])

        N_steps = int(simulation_time/ abs(timestep)) + 1
        
        state = np.array(initial_state, dtype=float)
        dim = state.size
        t_current = t0

        t_values = np.empty(N_steps)
        states = np.empty((N_steps, dim))

        while True:
        # for i in range(N_steps):
            t_values[i] = t_current
            states[i, :] = state

            logging.info(f"STEP {i} | t: {t_values[i]:.4f}  |  state: {state}")

            state = self.step(state, t_current, timestep, params)

            t_current += timestep

            if i % 10000 == 0 and on_save is not None:
                on_save(i, t_values.copy(), states.copy())

            if on_end_condition(t_current, state):
                break

        return t_values, states

    # def integrate_conditional(self, condition, timestep, initial_state, params):
    #     state = np.array(initial_state, dtype=float)
    #     dim = state.size
    #     # t_current - 0.0
        
    #     t_values = []
    #     states = []

    #     i = 0
    #     while True:
    #         #I would probably have to incorporate the stoping parameter in the state
    #         t_values.append(t_current)
    #         states.append(state.copy())


    #         logging.info(f"STEP {i} | t: {t_values[i]:.4f}  |  state: {state}")

    #         state = self.step(state, t_current, timestep, params)

    #         t_current += timestep
    #         i+= 1
    #         if condition(state):
    #             break
        
    #     return t_values, states

            