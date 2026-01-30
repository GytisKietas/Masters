import numpy as np

def function_compute(function, x_min, x_max, steps):
    x = np.linspace(x_min, x_max, steps)
    y = function(x)

    return np.column_stack(x, y)