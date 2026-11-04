"""Test of the role SA-I (team T12, storyline B, variant 2). Run: python -m pytest tests/test_SA-I.py   or   python tests/test_SA-I.py"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint4_generator_PUBLIC as gen
from src.meta import sa

INSTANCE_SEED, VARIANT = 12, 2


def _d(seed=INSTANCE_SEED):
    return gen.cp34_instance(seed, 'B', VARIANT)


def test_incremental_cost_equals_checker_on_1000_random_moves():
    d = _d()
    rng = random.Random(7)
    st = sa.State(d, VARIANT, sa.random_start(d, VARIANT, rng))
    n_applied = n_infeasible = n_swap = 0
    for _ in range(1000):
        mv = sa.propose(st, rng, 0.5)
        assert mv is not None
        delta, feas, data, moves = sa.evaluate_move(st, mv)
        cand = list(st.assign)
        for k, s in moves:
            cand[k] = s
        cost_c, problems = gen.evaluate_B(d, cand)
        assert abs((st.cost + delta) - cost_c) <= 1e-9, (mv, st.cost + delta, cost_c)
        cap_ok = not any('CPU' in p or 'memory' in p for p in problems)
        assert feas == cap_ok
        assert not any('zone' in p for p in problems)           # zone rule is never broken by a generated move
        n_swap += mv[0] == 'S'
        if feas:
            st.apply(moves, delta, data)
            n_applied += 1
            assert abs(st.cost - gen.evaluate_B(d, st.assign)[0]) <= 1e-9
        else:
            n_infeasible += 1
    assert n_applied > 100 and n_swap > 100                      # the test really exercises both move types


def test_random_start_is_feasible():
    d = _d()
    for s in range(20):
        a = sa.random_start(d, VARIANT, random.Random(s))
        assert gen.evaluate_B(d, a)[1] == []


def test_same_seed_same_run_and_other_seed_differs():
    d = _d()
    r1 = sa.solve(d, VARIANT, 3, 6000)
    r2 = sa.solve(d, VARIANT, 3, 6000)
    assert r1['solution'] == r2['solution'] and r1['cost'] == r2['cost'] and r1['trace'] == r2['trace']
    r3 = sa.solve(d, VARIANT, 4, 6000)
    assert r3['trace'] != r1['trace'] or r3['solution'] != r1['solution']


def test_exactly_the_budget_is_spent():
    d = _d()
    for budget in (2, 10, 500, 3000, 3500, 7000, 20000):
        assert sa.solve(d, VARIANT, 0, budget)['evals'] == budget, budget
    assert sa.solve(d, VARIANT, 0, 5000, params={'T0': 0.5, 'Tend': 0.001})['evals'] == 5000


def test_trace_never_increases_and_is_dense_enough():
    d = _d()
    r = sa.solve(d, VARIANT, 1, 12000)
    ev = [e for e, _ in r['trace']]
    cs = [c for _, c in r['trace']]
    assert ev == list(range(1000, 12001, 1000))
    assert all(b <= a + 1e-12 for a, b in zip(cs, cs[1:]))
    assert abs(cs[-1] - r['cost']) <= 1e-9


def test_reported_cost_and_flag_agree_with_checker_and_zones_hold():
    d = _d()
    r = sa.solve(d, VARIANT, 2, 8000)
    cost, problems = gen.evaluate_B(d, r['solution'])
    assert abs(cost - r['cost']) <= 1e-9
    assert r['feasible'] == (problems == [])
    assert r['feasible'] and problems == []


def test_start_argument_is_accepted_and_other_instance_runs():
    d = _d()
    start = sa.random_start(d, VARIANT, random.Random(5))
    r = sa.solve(d, VARIANT, 0, 4000, start=start)
    assert r['evals'] == 4000 and r['feasible']
    d2 = _d(99)                                                  # another instance of the same variant
    r2 = sa.solve(d2, VARIANT, 0, 4000)
    assert r2['evals'] == 4000 and r2['feasible'] == (gen.evaluate_B(d2, r2['solution'])[1] == [])


def test_does_not_use_global_random_state():
    d = _d()
    random.seed(1)
    a = sa.solve(d, VARIANT, 6, 4000)
    random.seed(2)
    b = sa.solve(d, VARIANT, 6, 4000)
    assert a['solution'] == b['solution'] and a['trace'] == b['trace']


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
