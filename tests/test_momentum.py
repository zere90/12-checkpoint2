"""Tests for optim/momentum.py -- M3 (Momentum) role.

1. R1 unit test from Section 3 (alpha=1e-3, beta=0.9 -> 3020).
2. H3 hand trace (f = 2x^2+4y^2, x0=(4,2), alpha=0.1, beta=0.6): two
   momentum iterates reproduced to 1e-3, and the plain-GD comparison.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optim.momentum import solve_momentum
from optim.gd import solve_gd
from problems import rosenbrock as R


def test_r1_unit_test_momentum():
    x, hist, k, status = solve_momentum(R.f, R.grad, R.R1_START, 1e-3, 0.9,
                                         stopping="absolute")
    assert status == "converged"
    assert k == 3020, f"expected 3020, got {k}"


def f_h3(x):
    return 2 * x[0] ** 2 + 4 * x[1] ** 2


def grad_h3(x):
    return np.array([4 * x[0], 8 * x[1]])


def test_h3_two_momentum_steps():
    x, hist, k, status = solve_momentum(f_h3, grad_h3, np.array([4.0, 2.0]),
                                         0.1, 0.6, stopping="absolute",
                                         cap=2)
    assert len(hist) >= 3
    assert np.allclose(hist[1], np.array([2.4, 0.4]), atol=1e-3)
    assert np.allclose(hist[2], np.array([0.48, -0.88]), atol=1e-3)
    assert abs(f_h3(hist[2]) - 3.5584) < 1e-3


def test_h3_momentum_vs_gd_same_start():
    x0 = np.array([4.0, 2.0])
    x, hist, k, status = solve_gd(f_h3, grad_h3, x0, 0.1,
                                   stopping="absolute", cap=2)
    gd_x2 = hist[-1]
    assert np.allclose(gd_x2, np.array([1.44, 0.08]), atol=1e-3)

    xm, histm, km, stm = solve_momentum(f_h3, grad_h3, x0, 0.1, 0.6,
                                         stopping="absolute", cap=2)
    mom_x2 = histm[-1]
    # momentum's x-coordinate has moved further toward 0 than GD's (faster
    # on the flatter/smaller-curvature direction after one extra step of
    # accumulated velocity); momentum's y-coordinate overshoots past 0,
    # GD's does not.
    assert abs(mom_x2[0]) < abs(gd_x2[0])
    assert mom_x2[1] < 0 < gd_x2[1]


if __name__ == "__main__":
    test_r1_unit_test_momentum()
    test_h3_two_momentum_steps()
    test_h3_momentum_vs_gd_same_start()
    print("test_momentum.py: all tests passed")
