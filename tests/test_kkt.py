"""tests/test_kkt.py -- M3 (Tamerlan). Reproduces the H3 hand trace with our own code.

H3(a): at x=(0.5,0.5,0.5) with pi=(2,3,3) (our task file), all five conditions hold exactly
(this IS the true LP optimum and its true dual, so every violation is 0).
H3(b): with pi'=(3,2,2), conditions (a)-(c) still hold (x unchanged, all rows still tight), but
reduced costs fail by 2 (on y3) and strong duality fails by a gap of 1.
H3(c): pi' still gives a valid lower bound (7) on z_LP (8), 1 below the tight bound.

Also: the required two-variable sanity check of the sign convention (Task 3's own instruction),
reused here as a regression test.
"""
import numpy as np
import _pathsetup  # noqa: F401
from model.build import build_triangle_model, LPModel
from lp.solve import solve_lp
from kkt.checker import check_conditions, check_model

TOL = 1e-6


def test_two_variable_sign_convention_sanity_check():
    # minimize x1+x2 s.t. x1+x2>=1, 0<=x<=1. Known answer: z=1, and the '>=' row's dual price is +1.
    toy = LPModel(
        c=np.array([1.0, 1.0]), A=np.array([[1.0, 1.0]]), b=np.array([1.0]), sense=['>='],
        lb=np.zeros(2), ub=np.ones(2), row_family=['toy'], row_label=['x1+x2>=1'],
        var_label=['x1', 'x2'], var_kind=['y', 'y'],
    )
    r = solve_lp(toy)
    assert abs(r.z - 1.0) < TOL
    assert abs(r.pi[0] - 1.0) < TOL          # '>=' row -> pi >= 0, and here exactly +1
    rep = check_model(toy, r)
    assert rep.all_satisfied


def test_h3a_true_dual_all_conditions_hold():
    m = build_triangle_model(5, 5, 6)
    x = np.array([0.5, 0.5, 0.5])
    pi = np.array([2.0, 3.0, 3.0])          # our task file's pi for H3(a)
    rep = check_conditions(m.c, m.A, m.b, m.sense, m.lb, m.ub, x, pi)
    assert rep.primal_feasibility < TOL
    assert rep.dual_feasibility < TOL
    assert rep.complementary_slackness < TOL
    assert rep.reduced_costs < TOL
    assert abs(rep.duality_gap) < TOL
    assert rep.all_satisfied


def test_h3b_wrong_dual_fails_d_and_e_only():
    m = build_triangle_model(5, 5, 6)
    x = np.array([0.5, 0.5, 0.5])
    pi_prime = np.array([3.0, 2.0, 2.0])    # our task file's pi' for H3(b)
    rep = check_conditions(m.c, m.A, m.b, m.sense, m.lb, m.ub, x, pi_prime)
    assert rep.primal_feasibility < TOL           # (a) unaffected by pi
    assert rep.dual_feasibility < TOL              # (b) pi' is still sign-feasible
    assert rep.complementary_slackness < TOL       # (c) all rows still tight regardless of pi
    assert abs(rep.reduced_costs - 2.0) < TOL       # (d) r3 = 6-(0+2+2) = 2
    assert abs(rep.duality_gap - 1.0) < TOL         # (e) 8 - (7+0) = 1
    assert not rep.all_satisfied


def test_h3c_pi_prime_still_a_valid_lower_bound():
    m = build_triangle_model(5, 5, 6)
    r = solve_lp(m)
    pi_prime = np.array([3.0, 2.0, 2.0])
    reduced = m.c - m.A.T @ pi_prime
    dual_value = float(m.b @ pi_prime + np.sum(np.minimum(0.0, reduced)))  # ub=1 for all 3 vars
    assert abs(dual_value - 7.0) < TOL
    assert dual_value <= r.z + TOL                  # weak duality: still a valid lower bound
    assert abs(r.z - dual_value - 1.0) < TOL         # 1 away from the tight bound (z_LP=8)


if __name__ == "__main__":
    test_two_variable_sign_convention_sanity_check()
    test_h3a_true_dual_all_conditions_hold()
    test_h3b_wrong_dual_fails_d_and_e_only()
    test_h3c_pi_prime_still_a_valid_lower_bound()
    print("test_kkt.py: all tests passed")
