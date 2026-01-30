import numpy as np
import logging

from utils import io

from RHS import slowroll
from RHS import phi
# from model import 

from integrators.rk4 import RK4Integrator

logger = logging.getLogger(__name__)

def run(args, ic):

    header = ["N", "phi", "phi_prime"]

    init = ic["initial_state"]

    return
