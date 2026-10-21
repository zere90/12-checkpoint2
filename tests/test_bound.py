"""tests/test_bound.py -- M5 (Zere, project lead). Reproduces the H5 hand trace with our own code.

H5: z_LP <= z* always; for the triangle problem z_LP=8, z*=10, and the gap (z-z_LP)/z is 0.2 at
the true optimum (1,1,0), (11-8)/11 at the feasible point (1,0,1), and 0.5 at the rounded-up
point (1,1,1).
"""
import numpy as np
import _pathsetup  # noqa: F401
from model.build import build_triangle_model
from lp.solve import solve_lp
from bound import gap, brute_force_binary_optimum

TOL = 1e-6


def test_h5a_z_lp_le_z_star():
    m = build_triangle_model(5, 5, 6)
    r = solve_lp(m)
    z_star, _ = brute_force_binary_optimum(m)
    assert z_star is not None
    assert r.z <= z_star + TOL


def test_h5b_three_gaps():
    m = build_triangle_model(5, 5, 6)
    r = solve_lp(m)
    z_lp = r.z
    z_star, best_point = brute_force_binary_optimum(m)
    assert abs(z_star - 10.0) < TOL
    assert tuple(best_point) == (1.0, 1.0, 0.0)

    cost_101 = float(m.c @ np.array([1.0, 0.0, 1.0]))     # 11
    cost_111 = float(m.c @ np.array([1.0, 1.0, 1.0]))     # 16

    assert abs(gap(z_star, z_lp) - 0.2) < TOL
    assert abs(gap(cost_101, z_lp) - (11 - 8) / 11) < TOL
    assert abs(gap(cost_111, z_lp) - 0.5) < TOL


def test_h5c_z_lp_le_z_star_on_real_instance_too():
    """Sanity extension beyond the triangle toy: the same z_LP<=z* inequality must hold for our
    real B2 instance once Checkpoint 4 produces an integer z (not computed here -- this just
    checks the LP relaxation runs and returns a finite lower bound, ready for that comparison)."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from checkpoint3_generator_PUBLIC import cp34_instance
    from model.build import build_b2_model
    d = cp34_instance(12, 'B', 2)
    bm = build_b2_model(d)
    r = solve_lp(bm)
    assert np.isfinite(r.z)
    assert abs(r.z - 5.209694989106753) < 1e-6


if __name__ == "__main__":
    test_h5a_z_lp_le_z_star()
    test_h5b_three_gaps()
    test_h5c_z_lp_le_z_star_on_real_instance_too()
    print("test_bound.py: all tests passed")
