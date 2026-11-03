"""tests/test_ga_a.py -- GA-A's own role test (Zere Zhussupbekova).

Guards exactly the four properties the CP4 role table requires for GA-A:
  1. same seed gives the same run;
  2. exactly the budget is spent;
  3. elitism keeps the best individual;
  4. the tournament never returns the worst of its entrants.

(1) and (2) are end-to-end checks of solve(). (3) and (4) are direct unit tests of
ga.py's own internal mechanics (_elite_of / _advance_generation / _tournament), not just
of the final output, so a broken elitism or tournament that happens to still produce a
plausible-looking final answer would still be caught.

These tests exercise src.meta.ga_problem -- GA-P's (Kabdulla Danial's) file -- through
the real agreed interface (random_individual, evaluate, crossover, mutate); nothing here
hard-codes the current contents of that file, so it will keep working once his real
implementation replaces whatever stub it starts from.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pathsetup  # noqa: F401 -- shared repo-root/src on sys.path, team convention

import numpy as np

import checkpoint4_generator_PUBLIC as gen
from src.meta.ga import _advance_generation, _elite_of, _tournament, solve
from src.meta.interface import check

D = gen.cp34_instance(12, "B", 2)
VARIANT = 2


def test_same_seed_same_run():
    r1 = solve(D, VARIANT, seed=7, budget=1500)
    r2 = solve(D, VARIANT, seed=7, budget=1500)
    assert r1["solution"] == r2["solution"]
    assert r1["cost"] == r2["cost"]
    assert r1["feasible"] == r2["feasible"]
    assert r1["trace"] == r2["trace"]
    print("test_same_seed_same_run: OK")


def test_exact_budget_spent():
    for budget in (1, 2, 5, 29, 30, 31, 59, 1000, 3333):
        r = solve(D, VARIANT, seed=1, budget=budget)
        assert r["evals"] == budget, f"budget {budget}: evals {r['evals']}"
    print("test_exact_budget_spent: OK")


def test_reported_result_matches_the_real_checker():
    r = solve(D, VARIANT, seed=2, budget=1500)
    cost, problems = check(D, VARIANT, r["solution"])
    assert abs(cost - r["cost"]) < 1e-6
    assert (len(problems) == 0) == r["feasible"]
    print("test_reported_result_matches_the_real_checker: OK")


def test_elitism_keeps_the_best_individual():
    """Build a small fake population with known, controlled fitness values (not drawn
    from random_individual), run one real generation step, and check that the single
    best (lowest-fitness) row of the OLD population reappears, completely unchanged, as
    the first entry of the NEW population -- this is elitism, tested directly rather
    than inferred from a final cost number."""
    rng = np.random.default_rng(123)
    # build individuals through the real interface so crossover/mutate get valid input
    import src.meta.ga as ga_mod

    pop = []
    for _ in range(10):
        ind = ga_mod.problem.random_individual(D, VARIANT, rng)
        fitness, cost, feasible = ga_mod.problem.evaluate(D, VARIANT, ind)
        pop.append((fitness, cost, feasible, ind))

    true_elite = _elite_of(pop)
    assert true_elite[0] == min(t[0] for t in pop)

    new_pop, evaluated = _advance_generation(D, VARIANT, pop, rng, p_c=0.9, N=10, evals_allowed=9)
    assert new_pop[0] is true_elite, "elitism must carry the exact elite row forward unchanged"
    assert new_pop[0][0] == min(t[0] for t in pop), "the carried-forward individual must be the best one"
    assert len(evaluated) == 9, "elitism's copy must not count as an evaluation"
    print("test_elitism_keeps_the_best_individual: OK")


class _ScriptedRNG:
    """A fake rng whose .integers(0, n, size=2) replays a fixed list of (i, j) pairs, so
    the test can force every possible draw (including i == j, the 'both entrants are the
    same individual' case) instead of hoping enough random draws happen to cover it."""

    def __init__(self, pairs):
        self._pairs = list(pairs)
        self._k = 0

    def integers(self, lo, hi, size=2):
        pair = self._pairs[self._k % len(self._pairs)]
        self._k += 1
        return np.array(pair)


def test_tournament_never_returns_the_worst_entrant():
    """Direct check of _tournament with every possible (i, j) draw out of a 2-individual
    population forced explicitly: (0,1), (1,0) must each return the strictly-better
    entrant; (0,0), (1,1) (the same individual drawn twice) must return that individual,
    which is correct since neither entrant in that draw is 'worse' than the other."""
    good = (1.0, 1.0, True, np.array([0]))
    bad = (99.0, 99.0, False, np.array([1]))
    pop = [good, bad]

    rng = _ScriptedRNG([(0, 1), (1, 0)])
    for _ in range(10):
        winner = _tournament(pop, rng)
        assert winner is good[3], "tournament returned the strictly worse entrant"

    rng_same = _ScriptedRNG([(1, 1)])
    for _ in range(5):
        winner = _tournament(pop, rng_same)
        assert winner is bad[3], "when both entrants are identical, that individual must be returned"

    # tie: whichever is returned, neither is "the worst" since both are equally good
    tie_a = (5.0, 5.0, True, np.array([10]))
    tie_b = (5.0, 5.0, True, np.array([11]))
    pop_tie = [tie_a, tie_b]
    rng_tie = _ScriptedRNG([(0, 1), (1, 0)])
    for _ in range(10):
        winner = _tournament(pop_tie, rng_tie)
        assert winner is tie_a[3] or winner is tie_b[3]
    print("test_tournament_never_returns_the_worst_entrant: OK")


if __name__ == "__main__":
    test_same_seed_same_run()
    test_exact_budget_spent()
    test_reported_result_matches_the_real_checker()
    test_elitism_keeps_the_best_individual()
    test_tournament_never_returns_the_worst_entrant()
    print("test_ga_a.py: all tests passed")
