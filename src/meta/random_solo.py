"""Random-search baseline of the SA-solo role: independent random assignments, same budget and seeds as SA.
Every drawn assignment is one evaluation, feasible or not. Each VM is drawn uniformly from its allowed servers."""
import random

from src.meta.interface import check
from src.meta.sa_solo import Problem


def solve(d, variant, seed, budget, params=None, start=None):
    P = Problem(d, variant)
    rng = random.Random(seed)
    r = rng.random
    best_assign, best_cost = None, float('inf')
    least, least_over = None, None
    feasible_draws = 0
    trace = []
    for e in range(1, int(budget) + 1):
        assign = [servers[int(r() * len(servers))] for servers in P.allowed]
        cpu, mem = P.loads(assign)
        over = P.overload(cpu, mem)
        if over == 0:
            feasible_draws += 1
            cost = P.cost(cpu)
            if cost < best_cost:
                best_cost, best_assign = cost, assign
        elif best_assign is None and (least is None or over < least_over):
            least, least_over = assign, over
        if e == 1 or e % 1000 == 0:
            trace.append((e, best_cost))
    if not trace or trace[-1][0] != int(budget):
        trace.append((int(budget), best_cost))
    final = best_assign if best_assign is not None else least
    checked_cost, problems = check(d, variant, list(final))
    return {'solution': [int(x) for x in final], 'cost': float(checked_cost), 'evals': int(budget),
            'feasible': len(problems) == 0, 'trace': trace, 'diag': {'feasible_draws': feasible_draws}}
