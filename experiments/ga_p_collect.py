"""Replay the two GA-A solvers with the starter runner and capture GA-P diagnostics."""

import argparse
import csv
import json
import math
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

from src.meta import ga, ga_problem

from experiments import run_final


def diversity(pop):
    return len({tuple(row[3]) for row in pop})


def run_seed(job):
    seed, directory = job
    destination = Path(directory) / f"{seed}.csv"
    detail = {"seed": seed}
    count = 0
    best = math.inf
    advance = ga._advance_generation
    solve = ga.solve

    def observe(rows):
        nonlocal count, best
        for _, cost, feasible, _ in rows:
            count += 1
            if feasible and cost < best:
                best = cost
                detail["last_improvement"] = count

    def record_generation(d, variant, population, rng, p_c, N, evals_allowed):
        if not count:
            detail["initial_distinct"] = diversity(population)
            observe(population)
        new, evaluated = advance(d, variant, population, rng, p_c, N, evals_allowed)
        observe(evaluated)
        if count == N + N - 1:
            detail["first_offspring_generation_distinct"] = diversity(new)
        if evals_allowed <= 2 * (N - 1):
            if len(new) == N:
                detail["last_full_distinct"] = diversity(new)
            if evals_allowed <= N - 1:
                detail["last_distinct"] = diversity(new)
                detail["last_size"] = len(new)
        return new, evaluated

    def observed_solve(*args, **kwargs):
        result = solve(*args, **kwargs)
        if result["evals"] != count or count != 100_000:
            raise ValueError("GA observation count disagrees with the protocol")
        detail.update(
            solution=result["solution"],
            cost=result["cost"],
            trace=[[x, y if math.isfinite(y) else None] for x, y in result["trace"]],
        )
        return result

    for name, configuration in [("ga", "GA, penalty"), ("random_search", "random")]:
        argv = [
            "run_final",
            "--module",
            f"src.meta.{name}",
            "--instance-seed",
            "12",
            "--story",
            "B",
            "--variant",
            "2",
            "--team",
            "T12",
            "--role",
            "GA-A",
            "--configuration",
            configuration,
            "--seeds",
            str(seed),
            "--out",
            str(destination),
        ]
        if name == "ga":
            settings = dict(ga.DEFAULT_PARAMS, rho=ga_problem.RHO)
            argv.extend(["--params", json.dumps(settings)])
        else:
            argv.append("--append")
        with (
            patch.object(sys, "argv", argv),
            patch.object(ga, "_advance_generation", record_generation),
            patch.object(ga, "solve", observed_solve),
        ):
            run_final.main()
    with destination.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2 or any(int(r["evals"]) != 100_000 for r in rows):
        raise ValueError("incomplete pair run")
    detail["pair_rows"] = rows
    return rows, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.out_dir.mkdir(exist_ok=True, parents=True)
    pair = args.out_dir / "Zhussupbekova_runs.csv"
    details = args.out_dir / "kabdulla_ga_details.json"
    if pair.exists() or details.exists():
        parser.error("use a fresh --out-dir to preserve existing pair results")
    if args.workers < 1:
        parser.error("--workers must be positive")
    with tempfile.TemporaryDirectory(dir=args.out_dir) as temp:
        jobs = [(seed, temp) for seed in range(10)]
        if args.workers == 1:
            data = list(map(run_seed, jobs))
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, 10)) as pool:
                data = list(pool.map(run_seed, jobs))
    with pair.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=run_final.COLS)
        writer.writeheader()
        for method in (0, 1):
            writer.writerows(rows[method] for rows, _ in data)
    details.write_text(
        json.dumps([detail for _, detail in data], allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
