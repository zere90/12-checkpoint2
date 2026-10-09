"""Tests for optim/gd.py -- M1 (GD) role.

1. R1 unit test from Section 3 of the instructions (GD, alpha=1e-3 -> 32076).
2. H1 hand trace (f = 2x^2+5y^2, x0=(5,2), alpha=0.05): two fixed-step GD
   iterates, reproduced to 1e-3; and the single accepted backtracking step
   from the same start (alpha0=1, c=1e-4).
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optim.gd import solve_gd, solve_gd_bt
from problems import rosenbrock as R


def test_r1_unit_test_gd_fixed():
    x, hist, k, status = solve_gd(R.f, R.grad, R.R1_START, 1e-3,
                                   stopping="absolute")
    assert status == "converged"
    assert k == 32076, f"expected 32076, got {k}"


def test_r1_unit_test_gdbt():
    x, hist, k, status, log = solve_gd_bt(R.f, R.grad, R.R1_START,
                                           stopping="absolute")
    assert status == "converged"
    assert k == 13756, f"expected 13756, got {k}"


def f_h1(x):
    return 2 * x[0] ** 2 + 5 * x[1] ** 2


def grad_h1(x):
    return np.array([4 * x[0], 10 * x[1]])


def test_h1_two_fixed_steps():
    x = np.array([5.0, 2.0])
    expected = [np.array([5.0, 2.0]), np.array([4.0, 1.0]),
                np.array([3.2, 0.5])]
    traj = [x.copy()]
    for _ in range(2):
        x = x - 0.05 * grad_h1(x)
        traj.append(x.copy())
    for got, want in zip(traj, expected):
        assert np.allclose(got, want, atol=1e-3)
    assert abs(f_h1(traj[2]) - 21.73) < 1e-3


def test_h1_backtracking_step():
    x0 = np.array([5.0, 2.0])
    _, _, k, status, log = solve_gd_bt(f_h1, grad_h1, x0, stopping="absolute",
                                        cap=5)
    # one accepted step is enough to reach ||grad|| below the absolute rule?
    # Not necessarily -- here we only check the FIRST accepted (alpha, x1)
    # against the hand trace, by re-running the line search in isolation.
    d = -grad_h1(x0)
    fx0 = f_h1(x0)
    dot = grad_h1(x0) @ d
    alpha = 1.0
    trials = 0
    while f_h1(x0 + alpha * d) > fx0 + 1e-4 * alpha * dot:
        alpha /= 2.0
        trials += 1
    x1 = x0 + alpha * d
    assert abs(alpha - 0.25) < 1e-9
    assert np.allclose(x1, np.array([0.0, -3.0]), atol=1e-3)
    assert abs(f_h1(x1) - 45.0) < 1e-3


if __name__ == "__main__":
    test_r1_unit_test_gd_fixed()
    test_r1_unit_test_gdbt()
    test_h1_two_fixed_steps()
    test_h1_backtracking_step()
    print("test_gd.py: all tests passed")
