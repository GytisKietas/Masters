import numpy as np
from .base import BaseIntegrator 

class RK4Integrator(BaseIntegrator):
    def __init__(self, rhs):
        self.rhs = rhs
        super().__init__()
    
    def step(self, state, t_current, timestep, params):
        k1 = self.rhs(t_current, state, params)
        k2 = self.rhs(t_current + 0.5*timestep, state + 0.5*timestep*k1, params)
        k3 = self.rhs(t_current + 0.5*timestep, state + 0.5*timestep*k2, params)
        k4 = self.rhs(t_current + timestep, state + timestep*k3, params)

        return state + (timestep/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    def integrate(self, simulation_time, timestep, initial_state, params):
        return super().integrate(simulation_time, timestep, initial_state, params)