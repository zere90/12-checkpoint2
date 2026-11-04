"""src/meta/ga.py -- GA-A (Zere Zhussupbekova): the genetic-algorithm loop.

Owns: tournament selection, elitism, generations, exact-budget accounting,
solve(...), the 10 final runs (experiments/run_final.py, already in the
starter kit), and the random-search baseline (src/meta/random_search.py).

Does NOT own: the representation, random_individual, evaluate (fitness,
cost, feasible), crossover, or mutate -- those are GA-P's (Kabdulla
Danial's) src/meta/ga_problem.py, agreed between the two of us at the
start (CP4 Instructions, Section 2) and imported here as `problem`.

Configuration: "GA, penalty" (team T12 task file). Offspring may be
infeasible; the fitness returned by problem.evaluate already contains
the penalty term. This module only ever selects/compares individuals by
fitness -- the *reported* solution/cost/feasible of solve() always come
from the real checker (via problem.evaluate's cost/feasible, which wrap
evaluate_B), never from the fitness number itself.

Rule-of-thumb GA parameters (CP4 Instructions, Section 4): population
size N=30, crossover probability p_c=0.9, mutation of 1 or 2 components
per offspring (handled inside problem.mutate), tournament of size 2,
elitism of 1 individual. All overridable via `params`, used for the
population-size experiment (this role's required analysis item).
"""
import math

import numpy as np

from src.meta import ga_problem as problem

DEFAULT_PARAMS = {
    "N": 30,  # population size
    "p_c": 0.9,  # crossover probability
    "trace_every": 1000,  # at least every 1000 evaluations, per the interface spec
}


def _tournament(pop, rng):
    """Return the individual with the better (lower) fitness of two entrants.

    pop: list of (fitness, cost, feasible, individual) rows. Draws two
    indices uniformly at random and returns the winner's individual --
    never the worse one, by construction.
    """
    i, j = rng.integers(0, len(pop), size=2)
    a, b = pop[i], pop[j]
    return a[3] if a[0] <= b[0] else b[3]


def _elite_of(pop):
    """Return the single (fitness, cost, feasible, individual) row with the lowest fitness."""
    return min(pop, key=lambda t: t[0])


def _make_offspring(d, variant, pop, rng, p_c):
    """Produce one not-yet-evaluated offspring.

    Two tournament winners -> crossover with probability p_c, otherwise an
    unchanged copy of one parent -> mutate.
    """
    parent1 = _tournament(pop, rng)
    parent2 = _tournament(pop, rng)
    if rng.random() < p_c:
        child = problem.crossover(parent1, parent2, rng)
    else:
        # An unchanged copy of a parent still costs one evaluation once it
        # is scored below (Section 3, "Evaluation" rule) -- we must not
        # reuse a cached cost for it.
        child = np.array(parent1, copy=True)
    return problem.mutate(d, variant, child, rng)


def _advance_generation(d, variant, pop, rng, p_c, N, evals_allowed):
    """Run one generation step.

    Keeps the current elite unchanged and free of cost as new_pop[0]
    (elitism of 1), then fills the rest of new_pop with up to N - 1
    freshly evaluated offspring, cut short after `evals_allowed`
    evaluations.

    Returns (new_pop, evaluated), where `evaluated` is the list of
    (fitness, cost, feasible, individual) rows actually scored during this
    call (elitism's free copy is not included), in the order they were
    produced -- each one is exactly one evaluation.
    """
    new_pop = [_elite_of(pop)]
    evaluated = []
    while len(new_pop) < N and len(evaluated) < evals_allowed:
        child = _make_offspring(d, variant, pop, rng, p_c)
        fitness, cost, feasible = problem.evaluate(d, variant, child)
        row = (fitness, cost, feasible, child)
        new_pop.append(row)
        evaluated.append(row)
    return new_pop, evaluated


def solve(d, variant, seed, budget, params=None, start=None):
    """GA, penalty configuration.

    See the module docstring and the CP4 interface spec for the exact
    meaning of every field of the returned dict.

    `start` is accepted per the interface but not used in Checkpoint 4 --
    every run starts from its own random population, as required.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    N = int(p["N"])
    p_c = float(p["p_c"])
    trace_every = int(p["trace_every"])

    rng = np.random.default_rng(seed)

    evals = 0
    trace = []
    best_feasible = None  # (cost, individual)
    best_by_fitness = None  # (fitness, cost, feasible, individual)

    def record(ind, fitness, cost, feasible):
        nonlocal best_feasible, best_by_fitness
        if best_by_fitness is None or fitness < best_by_fitness[0]:
            best_by_fitness = (fitness, cost, feasible, ind)
        if feasible and (best_feasible is None or cost < best_feasible[0]):
            best_feasible = (cost, ind)

    def maybe_trace():
        bf = best_feasible[0] if best_feasible is not None else math.inf
        if not trace or evals - trace[-1][0] >= trace_every or evals == budget:
            trace.append((evals, bf))

    # Initial population: every member counts as one evaluation (cut short
    # if the budget is smaller than N).
    pop = []
    for _ in range(N):
        if evals >= budget:
            break
        ind = problem.random_individual(d, variant, rng)
        fitness, cost, feasible = problem.evaluate(d, variant, ind)
        evals += 1
        record(ind, fitness, cost, feasible)
        pop.append((fitness, cost, feasible, ind))
        maybe_trace()

    # Generations: elitism (free) + offspring (1 evaluation each); the
    # last generation is cut short so evals lands exactly on budget.
    while evals < budget and pop:
        pop, evaluated = _advance_generation(d, variant, pop, rng, p_c, N, budget - evals)
        for fitness, cost, feasible, ind in evaluated:
            evals += 1
            record(ind, fitness, cost, feasible)
            maybe_trace()

    if not trace or trace[-1][0] != evals:
        maybe_trace()

    if best_feasible is not None:
        cost, ind = best_feasible
        feasible = True
    else:
        _, cost, feasible, ind = best_by_fitness
        feasible = False

    return {
        "solution": list(int(x) for x in ind),
        "cost": float(cost),
        "evals": evals,
        "feasible": bool(feasible),
        "trace": trace,
    }
