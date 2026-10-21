"""tests/test_lp.py -- M2 (Rakhima). Reproduces the H2 hand trace with our own code.

H2: y=(0.5,0.5,0.5) is LP-feasible, cost 8; this is also the true z_LP (we solve for it);
rounding up (1,1,1) costs 16 and is feasible, rounding down (0,0,0) is infeasible; the LP gap
(z*-z_LP)/z* = (10-8)/10 = 0.2, using z* = 10 from H1.
"""
import numpy as np
import _pathsetup  # noqa: F401
from model.build import build_triangle_model
from lp.solve import solve_lp

TOL = 1e-6
Z_STAR = 10.0  # integer optimum, from H1 / test_model.py


def test_half_point_is_feasible_and_costs_8():
    m = build_triangle_model(5, 5, 6)
    y = np.array([0.5, 0.5, 0.5])
    assert np.all(m.A @ y >= m.b - TOL)
    assert np.all((m.lb - TOL) <= y) and np.all(y <= (m.ub + TOL))
    cost = float(m.c @ y)
    assert abs(cost - 8.0) < TOL


def test_rounding():
    m = build_triangle_model(5, 5, 6)
    down = np.array([0.0, 0.0, 0.0])
    up = np.array([1.0, 1.0, 1.0])
    assert not np.all(m.A @ down >= m.b - TOL)              # rounding down: infeasible
    assert np.all(m.A @ up >= m.b - TOL)                     # rounding up: feasible
    assert abs(float(m.c @ up) - 16.0) < TOL


def test_z_lp_equals_8_and_gap():
    m = build_triangle_model(5, 5, 6)
    r = solve_lp(m)
    assert abs(r.z - 8.0) < TOL
    assert np.allclose(r.x, [0.5, 0.5, 0.5], atol=1e-6)
    gap = (Z_STAR - r.z) / Z_STAR
    assert abs(gap - 0.2) < TOL


if __name__ == "__main__":
    test_half_point_is_feasible_and_costs_8()
    test_rounding()
    test_z_lp_equals_8_and_gap()
    print("test_lp.py: all tests passed")
