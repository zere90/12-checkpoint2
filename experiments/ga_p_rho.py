"""GA-P's rho comparison using GA-A's existing solve, without editing ga.py.

Default seeds 100--104 are tuning seeds. Do not choose rho using final seeds
0--9. The initial population is assumed to be the first N evaluations, per
the agreed GA protocol. N comes from the settings used by solve().
"""

import argparse
import csv
import importlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint4_generator_PUBLIC import cp34_instance, evaluate_B
from src.meta import ga_problem as problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="src.meta.ga")
    parser.add_argument("--rhos", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 105)))
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--params", default="{}", help="JSON params for GA-A solve")
    parser.add_argument("--out", default="results/kabdulla_exp.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(set(args.rhos)) < 3 or any(not np.isfinite(r) or r <= 0 for r in args.rhos):
        parser.error("provide at least three distinct finite positive rho values")
    if len(set(args.rhos)) != len(args.rhos):
        parser.error("rho values must be distinct")
    if not args.seeds or any(s not in range(100, 105) for s in args.seeds):
        parser.error("parameter comparison must use only tuning seeds 100--104")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    params = json.loads(args.params)
    if not isinstance(params, dict):
        parser.error("--params must be a JSON object")
    if "rho" in params or "RHO" in params:
        parser.error("set rho through --rhos, not --params")
    try:
        ga = importlib.import_module(args.module)
    except ModuleNotFoundError as exc:
        parser.exit(2, f"Cannot import {args.module}: {exc}. Add GA-A's committed module first.\n")
    if getattr(ga, "problem", None) is not problem:
        parser.exit(2, "GA-A must import src.meta.ga_problem as problem and call problem.evaluate.\n")
    initial_population = int(params.get("N", ga.DEFAULT_PARAMS["N"]))
    if not 2 <= initial_population < args.budget:
        parser.error("budget must exceed the population size, which must be at least 2")

    output = Path(args.out)
    summary_path = output.with_name(output.stem + "_summary.csv")
    if not args.overwrite and (output.exists() or summary_path.exists()):
        parser.error("output already exists; use another --out or explicitly --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    d = cp34_instance(12, "B", 2)
    original_rho, original_evaluate = problem.RHO, problem.evaluate
    fields = ["team", "role", "configuration", "seed", "budget", "cost", "evals",
              "feasible", "seconds", "params", "rho", "offspring_evaluated",
              "feasible_offspring", "feasible_offspring_share"]
    rows = []
    try:
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for rho in args.rhos:
                for seed in args.seeds:
                    problem.RHO = rho
                    calls = offspring = feasible_offspring = 0

                    def observed_evaluate(data, variant, individual):
                        nonlocal calls, offspring, feasible_offspring
                        result = original_evaluate(data, variant, individual)
                        calls += 1
                        if calls > initial_population:
                            offspring += 1
                            feasible_offspring += int(result[2])
                        return result

                    problem.evaluate = observed_evaluate
                    started = time.perf_counter()
                    try:
                        result = ga.solve(d, 2, seed, args.budget, params=dict(params))
                    finally:
                        elapsed = time.perf_counter() - started
                        problem.evaluate = original_evaluate
                    if calls != args.budget or result["evals"] != args.budget:
                        raise RuntimeError("GA evaluation count disagrees with budget; check GA-A integration")
                    # This independent post-run check is not a search evaluation.
                    if result["solution"] is not None:
                        cost, errors = evaluate_B(d, result["solution"])
                        if bool(result["feasible"]) != (not bool(errors)):
                            raise RuntimeError("reported feasibility disagrees with checker")
                        if not np.isclose(cost, result["cost"], rtol=0, atol=1e-9, equal_nan=True):
                            raise RuntimeError("reported cost disagrees with checker")
                    elif result["feasible"]:
                        raise RuntimeError("GA reports a feasible result without a solution")
                    if problem.RHO != rho:
                        raise RuntimeError("GA changed RHO during solve; agree on one penalty configuration path")
                    row = dict(team="T12", role="GA-P", configuration="GA, penalty",
                               seed=seed, budget=args.budget, cost=result["cost"],
                               evals=result["evals"], feasible=bool(result["feasible"]),
                               seconds=elapsed,
                               params=json.dumps(dict(ga.DEFAULT_PARAMS, **params) | {"rho": rho},
                                                 sort_keys=True),
                               rho=rho, offspring_evaluated=offspring,
                               feasible_offspring=feasible_offspring,
                               feasible_offspring_share=feasible_offspring / offspring)
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
                    print(f"rho={rho:g} seed={seed}: cost={result['cost']}, "
                          f"feasible={result['feasible']}, offspring feasible="
                          f"{row['feasible_offspring_share']:.3%}", flush=True)
    finally:
        problem.RHO, problem.evaluate = original_rho, original_evaluate

    summary_fields = ["rho", "runs", "feasible_offspring_share", "mean_feasible_cost",
                      "best_feasible_cost", "worst_feasible_cost", "infeasible_run_share"]
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for rho in args.rhos:
            subset = [r for r in rows if r["rho"] == rho]
            costs = [r["cost"] for r in subset if r["feasible"]]
            writer.writerow(dict(rho=rho, runs=len(subset),
                feasible_offspring_share=sum(r["feasible_offspring"] for r in subset)
                    / sum(r["offspring_evaluated"] for r in subset),
                mean_feasible_cost=statistics.mean(costs) if costs else "",
                best_feasible_cost=min(costs) if costs else "",
                worst_feasible_cost=max(costs) if costs else "",
                infeasible_run_share=sum(not r["feasible"] for r in subset) / len(subset)))
    print(f"Wrote {output} and {summary_path}")


if __name__ == "__main__":
    main()
