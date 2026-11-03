"""experiments/run_popsize_diversity.py -- GA-A's own required analysis item (CP4 role
table): population diversity against the generation, and the effect of the population
size N (two values) on the final cost. Owned and authored by Zere Zhussupbekova (GA-A).

Re-uses ga.py's own internal generation step (_advance_generation) so the search
behaviour is byte-for-byte identical to the real solve() -- this script only adds the
extra bookkeeping (diversity per generation) that solve()'s fixed return format has no
room for.

Writes two files:
  results/<surname>_diversity.csv  (N, generation, evals_so_far, n_distinct)
      one representative seed (--diversity-seed, default 0) run to completion for each
      N in --N, logging the number of distinct individuals in the population at every
      generation.
  results/<surname>_popsize.csv    (N, seed, cost, evals, feasible)
      every seed in --seeds, run to completion, for each N in --N -- the N-vs-final-cost
      comparison.

Usage (defaults match the final-run protocol: budget 100000, seeds 0-9):
  python -m experiments.run_popsize_diversity --instance-seed 12 --story B --variant 2 \
      --N 15 30 --seeds 0-9 --budget 100000 --surname Zhussupbekova
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import checkpoint4_generator_PUBLIC as gen
import src.meta.ga as ga_mod


def parse_seeds(text):
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def _n_distinct(pop):
    return len(set(tuple(int(x) for x in row[3]) for row in pop))


def _solve_with_diversity(d, variant, seed, budget, N, p_c=0.9):
    """Same search as src.meta.ga.solve (same RNG stream, same _advance_generation
    calls), plus a (generation, evals_so_far, n_distinct) row recorded once per
    generation, starting with generation 0 = the initial population."""
    rng = np.random.default_rng(seed)
    problem = ga_mod.problem
    evals = 0

    pop = []
    for _ in range(N):
        if evals >= budget:
            break
        ind = problem.random_individual(d, variant, rng)
        fitness, cost, feasible = problem.evaluate(d, variant, ind)
        evals += 1
        pop.append((fitness, cost, feasible, ind))

    diversity_rows = [(0, evals, _n_distinct(pop))]
    best_feasible = None
    for fitness, cost, feasible, ind in pop:
        if feasible and (best_feasible is None or cost < best_feasible[0]):
            best_feasible = (cost, ind)

    gen_idx = 0
    while evals < budget and pop:
        gen_idx += 1
        pop, evaluated = ga_mod._advance_generation(d, variant, pop, rng, p_c, N, budget - evals)
        for fitness, cost, feasible, ind in evaluated:
            evals += 1
            if feasible and (best_feasible is None or cost < best_feasible[0]):
                best_feasible = (cost, ind)
        diversity_rows.append((gen_idx, evals, _n_distinct(pop)))

    cost = best_feasible[0] if best_feasible is not None else float("nan")
    feasible = best_feasible is not None
    return cost, evals, feasible, diversity_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-seed", type=int, required=True)
    ap.add_argument("--story", required=True, choices=["A", "B"])
    ap.add_argument("--variant", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--N", type=int, nargs=2, required=True, help="exactly two population sizes to compare")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--budget", type=int, default=100000)
    ap.add_argument("--diversity-seed", type=int, default=0, help="which seed's run to log per-generation diversity for")
    ap.add_argument("--surname", required=True)
    a = ap.parse_args()

    d = gen.cp34_instance(a.instance_seed, a.story, a.variant, cap=a.cap)
    seeds = parse_seeds(a.seeds)

    popsize_path = os.path.join("results", f"{a.surname}_popsize.csv")
    diversity_path = os.path.join("results", f"{a.surname}_diversity.csv")
    os.makedirs("results", exist_ok=True)

    with open(popsize_path, "w", newline="") as f_pop, open(diversity_path, "w", newline="") as f_div:
        w_pop = csv.writer(f_pop)
        w_pop.writerow(["N", "seed", "cost", "evals", "feasible"])
        w_div = csv.writer(f_div)
        w_div.writerow(["N", "generation", "evals_so_far", "n_distinct"])

        for N in a.N:
            for seed in seeds:
                cost, evals, feasible, diversity_rows = _solve_with_diversity(d, a.variant, seed, a.budget, N)
                w_pop.writerow([N, seed, f"{cost:.9f}", evals, feasible])
                print(f"N={N} seed={seed}: cost {cost:.6f} feasible {feasible}")
                if seed == a.diversity_seed:
                    for gen_idx, evals_so_far, n_distinct in diversity_rows:
                        w_div.writerow([N, gen_idx, evals_so_far, n_distinct])

    print(f"wrote {popsize_path} and {diversity_path}")


if __name__ == "__main__":
    main()
