"""Tests for optim/newton.py -- M2 (Newton) role.

1. R1 unit tests from Section 3 (pure -> 6, damped -> 21).
2. H2 hand trace (f = x^2 + e^y - 4y, x0=(3,1)): two pure Newton steps
   reproduced to 1e-3, plus the damped full-step-acceptance check at k=0.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optim.newton import solve_newton_pure, solve_newton_damped
from problems import rosenbrock as R


def test_r1_unit_test_newton_pure():
    x, hist, k, status = solve_newton_pure(R.f, R.grad, R.hess, R.R1_START,
                                            stopping="absolute")
    assert status == "converged"
    assert k == 6, f"expected 6, got {k}"


def test_r1_unit_test_newton_damped():
    x, hist, k, status, log = solve_newton_damped(R.f, R.grad, R.hess,
                                                    R.R1_START,
                                                    stopping="absolute")
    assert status == "converged"
    assert k == 21, f"expected 21, got {k}"


def f_h2(x):
    return x[0] ** 2 + np.exp(x[1]) - 4 * x[1]


def grad_h2(x):
    return np.array([2 * x[0], np.exp(x[1]) - 4.0])


def hess_h2(x):
    return np.array([[2.0, 0.0], [0.0, np.exp(x[1])]])


def test_h2_two_pure_newton_steps():
    x = np.array([3.0, 1.0])
    expected = [np.array([3.0, 1.0]),
                np.array([0.0, 1.471518]),
                np.array([0.0, 1.389825])]
    traj = [x.copy()]
    for _ in range(2):
        g = grad_h2(x)
        H = hess_h2(x)
        p = np.linalg.solve(H, -g)
        x = x + p
        traj.append(x.copy())
    for got, want in zip(traj, expected):
        assert np.allclose(got, want, atol=1e-3)
    ystar = np.log(4.0)
    assert abs(traj[2][1] - ystar) < 4e-3  # |y^2 - y*| ~= 0.0035


def test_h2_damped_accepts_full_step_at_k0():
    x0 = np.array([3.0, 1.0])
    g0 = grad_h2(x0)
    H0 = hess_h2(x0)
    p0 = np.linalg.solve(H0, -g0)
    assert g0 @ p0 < 0  # descent direction, no gradient fallback needed
    fx0 = f_h2(x0)
    dot = g0 @ p0
    lhs = f_h2(x0 + 1.0 * p0)
    rhs = fx0 + 1e-4 * 1.0 * dot
    assert lhs <= rhs  # damped Newton accepts alpha=1 at k=0


def test_h2_monotone_decrease():
    x, hist, k, status = solve_newton_pure(f_h2, grad_h2, hess_h2,
                                            np.array([3.0, 1.0]),
                                            stopping="absolute", cap=5)
    fs = [f_h2(xi) for xi in hist]
    assert all(fs[i + 1] < fs[i] for i in range(len(fs) - 1))


if __name__ == "__main__":
    test_r1_unit_test_newton_pure()
    test_r1_unit_test_newton_damped()
    test_h2_two_pure_newton_steps()
    test_h2_damped_accepts_full_step_at_k0()
    test_h2_monotone_decrease()
    print("test_newton.py: all tests passed")
