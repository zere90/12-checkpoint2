"""Rosenbrock test problem (R1, R2).

f(x1, x2) = (1 - x1)^2 + 100*(x2 - x1^2)^2
"""
import numpy as np


def f(x):
    x1, x2 = x
    return (1.0 - x1) ** 2 + 100.0 * (x2 - x1 ** 2) ** 2


def grad(x):
    x1, x2 = x
    df1 = -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1 ** 2)
    df2 = 200.0 * (x2 - x1 ** 2)
    return np.array([df1, df2])


def hess(x):
    x1, x2 = x
    h11 = 2.0 - 400.0 * x2 + 1200.0 * x1 ** 2
    h12 = -400.0 * x1
    return np.array([[h11, h12], [h12, 200.0]])


# R1: shared start for every team.
R1_START = np.array([-1.2, 1.0])

# R2: team-specific start, from checkpoint2_params(12) -> x0_rosen.
R2_START = np.array([-1.77, 2.08])

MINIMIZER = np.array([1.0, 1.0])
F_MIN = 0.0
