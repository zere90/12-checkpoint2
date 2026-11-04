"""Build GA-P tables, F1 and the worst-seed checks from the pair's saved runs."""

import argparse
import csv
import json
import math
import statistics as st
from itertools import combinations
from pathlib import Path

from checkpoint4_generator_PUBLIC import cp34_instance, evaluate_B

LP, REF = 5.209695, 5.268322  # T12 task file. Only post-run reporting uses these.


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def stats(rows):
    if sorted(int(r["seed"]) for r in rows) != list(range(10)):
        raise ValueError("expected one result per seed 0-9")
    if any(int(r["evals"]) != 100_000 or int(r["budget"]) != 100_000 for r in rows):
        raise ValueError("all final runs must use exactly 100000 evaluations")
    costs = [float(r["cost"]) for r in rows if r["feasible"] == "True"]
    z = {
        "best": min(costs),
        "median": st.median(costs),
        "mean": st.mean(costs),
        "worst": max(costs),
    }
    return {
        "cost": z,
        "feasible": len(costs),
        "sd": st.stdev(costs),
        "seconds": st.mean(float(r["seconds"]) for r in rows),
        "gap_ref": {k: 100 * (v - REF) / REF for k, v in z.items()},
        "gap_lp": {k: 100 * (v - LP) / v for k, v in z.items()},
    }


def neighbourhood(d, a):
    start, errors = evaluate_B(d, a)
    if errors:
        raise ValueError(errors)

    def inspect(moves):
        counts = {
            "tested": 0,
            "feasible": 0,
            "improving": 0,
            "beyond_roundoff": 0,
            "best": None,
        }
        for label, candidate in moves:
            cost, errors = evaluate_B(d, candidate)
            counts["tested"] += 1
            if errors:
                continue
            counts["feasible"] += 1
            if cost < start:
                counts["improving"] += 1
                counts["beyond_roundoff"] += int(start - cost > 1e-12)
                if counts["best"] is None or cost < counts["best"]["cost"]:
                    counts["best"] = {"move": label, "cost": cost, "gain": start - cost}
        return counts

    def transfers():
        for k, old in enumerate(a):
            for server in range(d["S"]):
                if server != old:
                    child = a.copy()
                    child[k] = server
                    yield [k, old, server], child

    def swaps():
        for i, j in combinations(range(len(a)), 2):
            if a[i] != a[j]:
                child = a.copy()
                child[i], child[j] = child[j], child[i]
                yield [i, j, a[i], a[j]], child

    return {"transfer": inspect(transfers()), "swap": inspect(swaps())}


def figure(traces, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    grid = [x for x, _ in traces[0]]
    if any([x for x, _ in t] != grid for t in traces):
        raise ValueError("trace grids disagree")
    points = []
    for i, x in enumerate(grid):
        v = [t[i][1] for t in traces]
        if all(y is not None for y in v):
            points.append([x, min(v), st.mean(v), max(v)])
    xs, lo, mean, hi = zip(*points)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.05), width_ratios=(1, 1.4))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.78, wspace=0.36)
    for ax, start in zip(axes, (0, next(i for i, x in enumerate(xs) if x >= 10001))):
        ax.fill_between(
            xs[start:],
            lo[start:],
            hi[start:],
            step="post",
            color="#c8dce9",
            label="Min–max",
        )
        ax.step(
            xs[start:],
            mean[start:],
            where="post",
            color="#175273",
            lw=1.7,
            label="Mean best feasible",
        )
        ax.axhline(REF, color="#a34025", ls="--", lw=1.15, label="Exact optimum")
        ax.axhline(LP, color="#586473", ls=":", lw=1.5, label="LP bound")
        ax.set_xlim(xs[start], xs[-1])
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, _: str(int(x / 1000)) + "k" if x >= 1000 else str(int(x))
            )
        )
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Evaluations", fontsize=9)
    axes[0].set_title("Full trace", fontsize=10, loc="left")
    axes[1].set_title("After 10,000 evaluations", fontsize=10, loc="left")
    axes[0].set_ylabel("Checker cost", fontsize=9)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    root = parser.parse_args().results
    details = json.loads((root / "kabdulla_ga_details.json").read_text())
    pair_file = root / "Zhussupbekova_runs.csv"
    pair = (
        read_csv(pair_file)
        if pair_file.exists()
        else [row for entry in details for row in entry["pair_rows"]]
    )
    d = cp34_instance(12, "B", 2)
    ga = [r for r in pair if r["configuration"] == "GA, penalty"]
    random = [r for r in pair if r["configuration"] == "random"]
    random_by_seed = {int(row["seed"]): row for row in random}
    lookup = {int(r["seed"]): r for r in ga}
    if sorted(x["seed"] for x in details) != list(range(10)):
        raise ValueError("expected replay details for seeds 0–9")
    for entry in details:
        r = lookup[entry["seed"]]
        original_random = next(
            row for row in entry["pair_rows"] if row["configuration"] == "random"
        )
        current_random = random_by_seed[entry["seed"]]
        if original_random["feasible"] != current_random[
            "feasible"
        ] or not math.isclose(
            float(original_random["cost"]),
            float(current_random["cost"]),
            abs_tol=1e-9,
            rel_tol=0,
        ):
            raise ValueError(
                f"random baseline disagrees with replay at seed {entry['seed']}"
            )
        cost, errors = evaluate_B(d, entry["solution"])
        if not math.isclose(cost, float(r["cost"]), abs_tol=1e-9, rel_tol=0) or (
            not errors
        ) != (r["feasible"] == "True"):
            raise ValueError(f"pair CSV disagrees with replay at seed {entry['seed']}")
        if (not errors) and entry["trace"][-1] != [100_000, cost]:
            raise ValueError("trace endpoint is inconsistent")
    worst = max(
        (x for x in details if lookup[x["seed"]]["feasible"] == "True"),
        key=lambda x: x["cost"],
    )
    points = figure([entry["trace"] for entry in details], root / "F1_GA-P.png")
    summary = {
        "ga": stats(ga),
        "random": stats(random),
        "worst": {k: v for k, v in worst.items() if k not in ("trace", "pair_rows")},
        "moves": neighbourhood(d, worst["solution"]),
        "curve": points,
        "prediction_sweep": read_csv(root / "kabdulla_exp_summary.csv"),
    }
    (root / "kabdulla_analysis.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
