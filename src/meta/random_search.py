"""src/meta/random_search.py -- GA-A (Zere Zhussupbekova): the random-search baseline.

Owned by GA-A per the CP4 role table ("the 10 final runs ...; the
random-search baseline"). Same solve(...) interface as src/meta/ga.py and
the same budget/seed protocol, so that Question 6 of the analysis
("better than random search? how far from the bound?") is a fair,
apples-to-apples comparison: both draw candidates with
problem.random_individual and measure them with problem.evaluate / the
real checker, nothing else differs.

One evaluation = one freshly drawn random candidate; the starting
candidate counts too (Section 3, "Evaluation"). evals always equals
budget exactly -- cut short on the last one if needed (not applicable
here, since every evaluation is independent, but kept for symmetry with
ga.py and in case budget is 0).
"""
import math

import numpy as np

from src.meta import ga_problem as problem

DEFAULT_PARAMS = {
    "trace_every": 1000,  # at least every 1000 evaluations, per the interface spec
}


def solve(d, variant, seed, budget, params=None, start=None):
    """Draw `budget` independent random individuals and report the best feasible one.

    `start` is accepted per the interface but not used in Checkpoint 4.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    trace_every = int(p["trace_every"])

    rng = np.random.default_rng(seed)

    evals = 0
    trace = []
    best_feasible = None  # (cost, individual)
    best_by_fitness = None  # (fitness, cost, feasible, individual)

    def maybe_trace():
        bf = best_feasible[0] if best_feasible is not None else math.inf
        if not trace or evals - trace[-1][0] >= trace_every or evals == budget:
            trace.append((evals, bf))

    while evals < budget:
        ind = problem.random_individual(d, variant, rng)
        fitness, cost, feasible = problem.evaluate(d, variant, ind)
        evals += 1
        if best_by_fitness is None or fitness < best_by_fitness[0]:
            best_by_fitness = (fitness, cost, feasible, ind)
        if feasible and (best_feasible is None or cost < best_feasible[0]):
            best_feasible = (cost, ind)
        maybe_trace()

    if not trace or trace[-1][0] != evals:
        maybe_trace()

    if best_feasible is not None:
        cost, ind = best_feasible
        feasible = True
    else:
        if best_by_fitness is None:
            # budget == 0: no candidate was ever drawn
            return {"solution": None, "cost": float("nan"), "evals": 0, "feasible": False, "trace": []}
        _, cost, feasible, ind = best_by_fitness
        feasible = False

    return {
        "solution": list(int(x) for x in ind),
        "cost": float(cost),
        "evals": evals,
        "feasible": bool(feasible),
        "trace": trace,
    }
