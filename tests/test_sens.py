"""tests/test_sens.py -- M4 (Danial). Reproduces the H4 hand trace with our own code.

H4: raising the RHS of (T1) to 1+eps, with all three rows kept tight, gives y1=y2=0.5+eps/2,
y3=0.5-eps/2 and z(1+eps)=8+2*eps -- slope 2, matching pi1=2 from the task file. Valid for
-1 <= eps <= 1 (from 0<=y<=1). We check this both via the closed form and via src/sens itself.
"""
import numpy as np
import _pathsetup  # noqa: F401
from model.build import build_triangle_model
from sens.perturb import rhs_sensitivity
from lp.solve import solve_lp

TOL = 1e-6


def test_h4_closed_form_point_and_slope():
    for eps in (0.0, 0.01, 0.3, 0.99, -0.5):
        y1 = y2 = 0.5 + eps / 2
        y3 = 0.5 - eps / 2
        if not (-1 - TOL <= eps <= 1 + TOL):
            continue
        assert -TOL <= y1 <= 1 + TOL and -TOL <= y3 <= 1 + TOL
        z = 5 * y1 + 5 * y2 + 6 * y3
        assert abs(z - (8 + 2 * eps)) < TOL


def test_h4_via_src_sens_module():
    m = build_triangle_model(5, 5, 6)
    s = rhs_sensitivity(m, row_index=0, delta=0.01)   # row 0 = (T1)
    assert abs(s.left_slope - 2.0) < 1e-4
    assert abs(s.right_slope - 2.0) < 1e-4
    assert abs(s.pi - 2.0) < TOL
    assert s.interval_ok and not s.degenerate


def test_h4b_validity_boundary():
    # at eps=1, b=2: all-tight point is y1=y2=1, y3=0 -- still inside [0,1], z=10
    y = np.array([1.0, 1.0, 0.0])
    assert abs(5 * y[0] + 5 * y[1] + 6 * y[2] - 10.0) < TOL
    # at eps slightly above 1, y3 would go negative -- no longer valid
    eps = 1.01
    y3 = 0.5 - eps / 2
    assert y3 < 0


if __name__ == "__main__":
    test_h4_closed_form_point_and_slope()
    test_h4_via_src_sens_module()
    test_h4b_validity_boundary()
    print("test_sens.py: all tests passed")
