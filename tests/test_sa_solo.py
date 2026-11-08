"""Role test of SA-solo (the same guards as SA-I). Run: python tests/test_sa_solo.py  or  pytest tests/test_sa_solo.py"""
import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint4_generator_PUBLIC as gen
from src.meta import sa_solo
from src.meta.interface import check

SEEDS = [(12, 2), (3, 2), (7, 2)]


def instance(seed=12, variant=2):
    return gen.cp34_instance(seed, 'B', variant)


def test_incremental_cost_equals_checker_on_1000_moves():
    for inst, variant in SEEDS:
        d = instance(inst, variant)
        P = sa_solo.Problem(d, variant)
        rng = random.Random(1)
        assign = sa_solo.build_start(P, rng)
        cpu, mem = P.loads(assign)
        cost = check(d, variant, assign)[0]
        for _ in range(1000):
            k, new = sa_solo.propose(P, assign, rng.random)
            assert new != assign[k] and new in P.allowed[k]
            cost += sa_solo.transfer_delta(P, cpu, k, assign[k], new)
            sa_solo.apply_transfer(P, assign, cpu, mem, k, new)
            assert abs(cost - check(d, variant, assign)[0]) <= 1e-9


def test_tracked_cost_of_a_whole_run_equals_checker():
    d = instance()
    res = sa_solo.solve(d, 2, 5, 20000, params={'T0': 0.5, 'Tend': 0.001})
    assert abs(res['diag']['current_cost'] - check(d, 2, res['diag']['current_assign'])[0]) <= 1e-9


def test_same_seed_same_run_and_other_seed_differs():
    d = instance()
    a = sa_solo.solve(d, 2, 7, 5000)
    b = sa_solo.solve(d, 2, 7, 5000)
    c = sa_solo.solve(d, 2, 8, 5000)
    assert a['solution'] == b['solution'] and a['cost'] == b['cost'] and a['trace'] == b['trace']
    assert a['trace'] != c['trace']


def test_global_random_state_is_not_used():
    d = instance()
    random.seed(1)
    a = sa_solo.solve(d, 2, 3, 3000)
    random.seed(999)
    random.random()
    b = sa_solo.solve(d, 2, 3, 3000)
    assert a['trace'] == b['trace']


def test_exactly_the_budget_is_spent():
    d = instance()
    for budget in (1, 2, 50, 999, 1000, 1001, 4000, 30001):
        for params in (None, {'T0': 0.5, 'Tend': 0.001}):
            assert sa_solo.solve(d, 2, 0, budget, params=params)['evals'] == budget


def test_trace_format_never_increases_and_is_dense():
    d = instance()
    for params in (None, {'T0': 0.5, 'Tend': 0.001}):
        trace = sa_solo.solve(d, 2, 1, 12345, params=params)['trace']
        evals = [e for e, _ in trace]
        best = [b for _, b in trace]
        assert evals[-1] == 12345 and all(b - a <= 1000 for a, b in zip(evals, evals[1:]))
        assert all(y <= x for x, y in zip(best, best[1:]))


def test_result_agrees_with_checker():
    for inst, variant in SEEDS:
        d = instance(inst, variant)
        res = sa_solo.solve(d, variant, 2, 5000)
        cost, problems = check(d, variant, res['solution'])
        assert res['feasible'] == (len(problems) == 0)
        assert abs(res['cost'] - cost) <= 1e-9


def test_moves_respect_the_zones():
    d = instance()
    P = sa_solo.Problem(d, 2)
    for k in range(P.M):
        z = int(d['vm_required_zone'][k])
        assert all(z == -1 or int(d['zone'][l]) == z for l in P.allowed[k])


def test_transfer_only_cannot_lose_the_best_solution():
    d = instance()
    res = sa_solo.solve(d, 2, 4, 20000, params={'T0': 5.0, 'Tend': 1.0})   # hot: the final state is poor
    assert abs(res['cost'] - res['trace'][-1][1]) <= 1e-9
    assert res['cost'] <= res['diag']['current_cost'] + 1e-9


def test_impossible_instance_is_reported_as_infeasible():
    d = copy.deepcopy(instance())
    d['cpu_cap'] = d['cpu_cap'] // 4                                        # total CPU cannot fit
    res = sa_solo.solve(d, 2, 0, 3000)
    assert res['feasible'] is False and res['evals'] == 3000
    assert len(check(d, 2, res['solution'])[1]) > 0


def test_supplied_start_is_accepted():
    d = instance()
    P = sa_solo.Problem(d, 2)
    start = sa_solo.build_start(P, random.Random(9))
    res = sa_solo.solve(d, 2, 0, 2000, params={'T0': 1e-9, 'Tend': 1e-10}, start=start)
    assert res['feasible'] and res['cost'] <= check(d, 2, start)[0] + 1e-9


if __name__ == '__main__':
    tests = [v for n, v in sorted(globals().items()) if n.startswith('test_')]
    for t in tests:
        t()
        print('ok', t.__name__)
    print(len(tests), 'tests passed')
