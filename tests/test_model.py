"""tests/test_model.py -- M1 (Tilesbay). Reproduces the H1 hand trace with our own code.

H1: triangle problem, c=(5,5,6); matrix form c, A, b with 3 variables and 3 rows; list all 8
binary points, mark feasible, and find the integer optimum z* = 10 at y = (1,1,0).
"""
import itertools
import numpy as np
import _pathsetup  # noqa: F401
from model.build import build_triangle_model

TOL = 1e-6


def test_matrix_form():
    m = build_triangle_model(5, 5, 6)
    assert np.allclose(m.c, [5, 5, 6])
    assert np.allclose(m.A, [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    assert np.allclose(m.b, [1, 1, 1])
    assert m.sense == ['>=', '>=', '>=']
    assert m.n_vars == 3
    assert m.n_rows == 3


def test_b2_model_sizes_match_task_file():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from checkpoint3_generator_PUBLIC import cp34_instance
    from model.build import build_b2_model
    d = cp34_instance(12, 'B', 2)
    bm = build_b2_model(d)
    assert d['S'] == 10 and d['M'] == 35
    assert int(d['cpu_cap'].sum()) == 454
    assert int(d['mem_cap'].sum()) == 1680
    assert int(d['cpu_req'].sum()) == 172
    assert int(d['mem_req'].sum()) == 615
    assert bm.meta['n_forbidden'] == 60
    assert bm.n_vars == 360
    assert bm.n_rows == 95


def test_h1_all_eight_binary_points():
    m = build_triangle_model(5, 5, 6)
    best_cost, best_point, feasible_points = None, None, []
    for bits in itertools.product([0, 1], repeat=3):
        y = np.array(bits, dtype=float)
        feasible = np.all(m.A @ y >= m.b - TOL)
        if feasible:
            cost = float(m.c @ y)
            feasible_points.append((bits, cost))
            if best_cost is None or cost < best_cost:
                best_cost, best_point = cost, bits
    # feasible points, per H1(c): (0,1,1)=11, (1,0,1)=11, (1,1,0)=10, (1,1,1)=16
    feasible_set = dict(feasible_points)
    assert feasible_set[(0, 1, 1)] == 11
    assert feasible_set[(1, 0, 1)] == 11
    assert feasible_set[(1, 1, 0)] == 10
    assert feasible_set[(1, 1, 1)] == 16
    assert len(feasible_points) == 4
    assert best_point == (1, 1, 0)
    assert abs(best_cost - 10) < TOL


if __name__ == "__main__":
    test_matrix_form()
    test_b2_model_sizes_match_task_file()
    test_h1_all_eight_binary_points()
    print("test_model.py: all tests passed")
