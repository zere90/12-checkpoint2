"""Tests for optim/adam.py -- M4 (Adam) role.

1. R1 unit test from Section 3 (alpha=0.1, default betas/eps -> 1706).
2. H4 hand trace (f = 3x^2+5y^2, x0=(2,1), alpha=0.1): two bias-corrected
   Adam steps reproduced to 1e-3, and the coordinate-wise-scaling check.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optim.adam import solve_adam
from problems import rosenbrock as R


def test_r1_unit_test_adam():
    x, hist, k, status = solve_adam(R.f, R.grad, R.R1_START, 0.1,
                                     stopping="absolute")
    assert status == "converged"
    assert k == 1706, f"expected 1706, got {k}"


def f_h4(x):
    return 3 * x[0] ** 2 + 5 * x[1] ** 2


def grad_h4(x):
    return np.array([6 * x[0], 10 * x[1]])


def test_h4_two_adam_steps():
    x, hist, k, status = solve_adam(f_h4, grad_h4, np.array([2.0, 1.0]), 0.1,
                                     stopping="absolute", cap=2)
    assert len(hist) >= 3
    assert np.allclose(hist[1], np.array([1.9, 0.9]), atol=1e-3)
    assert np.allclose(hist[2], np.array([1.800166, 0.800412]), atol=1e-3)


def test_h4_coordinate_wise_scaling():
    """Despite grad = (12, 10) at x0 (ratio 1.2 : 1), Adam's first step is
    (0.1, 0.1) to within bias-correction round-off -- i.e. essentially
    alpha*sign(grad) in EVERY coordinate, not alpha*grad. Plain GD's first
    step would be alpha*grad = (1.2, 1.0): a 1.2x ratio between the two
    coordinates, same as the gradient. Adam removes that ratio almost
    entirely on step 1."""
    x0 = np.array([2.0, 1.0])
    g0 = grad_h4(x0)
    gd_step = 0.1 * g0
    x, hist, k, status = solve_adam(f_h4, grad_h4, x0, 0.1,
                                     stopping="absolute", cap=1)
    adam_step = x0 - hist[1]
    assert np.allclose(adam_step, np.array([0.1, 0.1]), atol=1e-3)
    assert not np.allclose(gd_step, adam_step, atol=1e-2)


if __name__ == "__main__":
    test_r1_unit_test_adam()
    test_h4_two_adam_steps()
    test_h4_coordinate_wise_scaling()
    print("test_adam.py: all tests passed")
