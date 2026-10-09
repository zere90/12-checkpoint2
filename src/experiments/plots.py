"""python -m experiments.plots

Owner: M3 -- Momentum role (5-member team: plots.py is M3's second file,
per Section 5 of the instructions).

Regenerates the three required figures from results/table1.csv and
results/table2.csv (which run_all.py must have produced first), using
the tuned hyper-parameters that are already on record there. Nothing
here is randomized: re-running always gives the same three PNGs.

    F1  trajectories of all six solvers on R1, over log-spaced contours
        (levels = np.logspace(-1, 3.5, 20))
    F2  convergence curves ||grad f|| vs k (semilogy) on Q1 and Q2,
        all six solvers
    F3  convergence curves on the project block, all six solvers
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from problems import rosenbrock as Rb
from problems import quadratic as Q
from problems import project as P
from optim.gd import solve_gd, solve_gd_bt
from optim.newton import solve_newton_pure, solve_newton_damped
from optim.momentum import solve_momentum
from optim.adam import solve_adam

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "..", "results",
                       "figures")
os.makedirs(FIGDIR, exist_ok=True)

SOLVER_STYLE = {
    "GD": dict(color="tab:blue", marker="o"),
    "GD-BT": dict(color="tab:orange", marker="s"),
    "Newton (pure)": dict(color="tab:green", marker="^"),
    "Newton (damped)": dict(color="tab:red", marker="v"),
    "Momentum": dict(color="tab:purple", marker="D"),
    "Adam": dict(color="tab:brown", marker="*"),
}


def _read_row(csv_path, problem_name):
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["problem"] == problem_name:
                return row
    raise KeyError(problem_name)


def _f(x):
    return float(x)


# ----------------------------------------------------------------------
# F1: trajectories on R1 over log-spaced contours
# ----------------------------------------------------------------------
def figure_f1():
    row = _read_row(os.path.join(RESULTS, "table1.csv"), "R1")
    x0 = Rb.R1_START

    hists = {}
    _, h, k, st = solve_gd(Rb.f, Rb.grad, x0, _f(row["GD_alpha"]),
                            stopping="absolute")
    hists["GD"] = h
    _, h, k, st, log = solve_gd_bt(Rb.f, Rb.grad, x0, stopping="absolute")
    hists["GD-BT"] = h
    _, h, k, st = solve_newton_pure(Rb.f, Rb.grad, Rb.hess, x0,
                                     stopping="absolute")
    hists["Newton (pure)"] = h
    _, h, k, st, log = solve_newton_damped(Rb.f, Rb.grad, Rb.hess, x0,
                                            stopping="absolute")
    hists["Newton (damped)"] = h
    _, h, k, st = solve_momentum(Rb.f, Rb.grad, x0, _f(row["Momentum_alpha"]),
                                  _f(row["Momentum_beta"]),
                                  stopping="absolute")
    hists["Momentum"] = h
    _, h, k, st = solve_adam(Rb.f, Rb.grad, x0, _f(row["Adam_alpha"]),
                              stopping="absolute")
    hists["Adam"] = h

    xs = np.linspace(-2.0, 2.0, 400)
    ys = np.linspace(-1.0, 3.0, 400)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = (1 - XX) ** 2 + 100 * (YY - XX ** 2) ** 2

    fig, ax = plt.subplots(figsize=(7, 6))
    levels = np.logspace(-1, 3.5, 20)
    ax.contour(XX, YY, ZZ, levels=levels, cmap="Greys", linewidths=0.6)

    for name, h in hists.items():
        arr = np.array(h)
        style = SOLVER_STYLE[name]
        # subsample very long trajectories for a readable marker count,
        # but always keep the first/last 5 points exactly
        n = len(arr)
        if n > 60:
            idx = np.unique(np.concatenate([
                np.arange(5), np.linspace(5, n - 6, 50).astype(int),
                np.arange(n - 5, n)]))
        else:
            idx = np.arange(n)
        ax.plot(arr[idx, 0], arr[idx, 1], "-", color=style["color"],
                linewidth=1.3, alpha=0.85,
                label=f"{name} (k={n - 1})")
        ax.plot(arr[idx, 0], arr[idx, 1], style["marker"],
                color=style["color"], markersize=3, alpha=0.6)

    ax.plot(1, 1, "k*", markersize=14, label="minimizer (1,1)")
    ax.plot(x0[0], x0[1], "kX", markersize=10, label="start")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("F1 -- R1 (Rosenbrock): trajectories of all six solvers")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "F1_r1_trajectories.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------
# F2: convergence curves ||grad|| vs k on Q1 and Q2, all six solvers
# ----------------------------------------------------------------------
def _grad_norms_q(h, grad_fn):
    return [np.linalg.norm(grad_fn(xi)) for xi in h]


def figure_f2():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (name_q, f, grad, hess, x0, label) in zip(
        axes,
        [("Q1", Q.q1_f, Q.q1_grad, Q.q1_hess, Q.Q1_START, "Q1"),
         ("Q2", Q.q2_f, Q.q2_grad, Q.q2_hess, Q.Q2_START, "Q2")]):

        row = _read_row(os.path.join(RESULTS, "table1.csv"), name_q)

        _, h, k, st = solve_gd(f, grad, x0, _f(row["GD_alpha"]),
                                stopping="relative")
        ax.semilogy(_grad_norms_q(h, grad), label=f"GD (k={k})",
                    color=SOLVER_STYLE["GD"]["color"])

        _, h, k, st, log = solve_gd_bt(f, grad, x0, stopping="relative")
        ax.semilogy(_grad_norms_q(h, grad), label=f"GD-BT (k={k})",
                    color=SOLVER_STYLE["GD-BT"]["color"])

        _, h, k, st = solve_newton_pure(f, grad, hess, x0,
                                         stopping="relative")
        ax.semilogy(_grad_norms_q(h, grad), label=f"Newton pure (k={k})",
                    color=SOLVER_STYLE["Newton (pure)"]["color"])

        _, h, k, st, log = solve_newton_damped(f, grad, hess, x0,
                                                stopping="relative")
        ax.semilogy(_grad_norms_q(h, grad), label=f"Newton damped (k={k})",
                    color=SOLVER_STYLE["Newton (damped)"]["color"])

        _, h, k, st = solve_momentum(f, grad, x0, _f(row["Momentum_alpha"]),
                                      _f(row["Momentum_beta"]),
                                      stopping="relative")
        ax.semilogy(_grad_norms_q(h, grad), label=f"Momentum (k={k})",
                    color=SOLVER_STYLE["Momentum"]["color"])

        _, h, k, st = solve_adam(f, grad, x0, _f(row["Adam_alpha"]),
                                  stopping="relative")
        ax.semilogy(_grad_norms_q(h, grad), label=f"Adam (k={k})",
                    color=SOLVER_STYLE["Adam"]["color"])

        ax.set_title(f"{label} (c={Q.C}, " +
                     ("axis-aligned" if label == "Q1" else
                      f"rotated {Q.THETA_DEG}deg") + ")")
        ax.set_xlabel("iteration $k$")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"$\|\nabla f(x_k)\|$")
    fig.suptitle("F2 -- convergence on Q1 vs Q2 (tuned hyper-parameters)")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "F2_q1_q2_convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------
# F3: convergence curves on the project block, all six solvers
# ----------------------------------------------------------------------
def figure_f3():
    row = _read_row(os.path.join(RESULTS, "table2.csv"),
                     "Project (storyline B)")
    x0 = P.START
    fig, ax = plt.subplots(figsize=(7, 5))

    _, h, k, st = solve_gd(P.f, P.grad, x0, _f(row["GD_alpha"]),
                            stopping="relative")
    ax.semilogy([np.linalg.norm(P.grad(xi)) for xi in h],
                label=f"GD, alpha={_f(row['GD_alpha']):.3g} (k={k})",
                color=SOLVER_STYLE["GD"]["color"])

    _, h, k, st, log = solve_gd_bt(P.f, P.grad, x0, stopping="relative")
    ax.semilogy([np.linalg.norm(P.grad(xi)) for xi in h],
                label=f"GD-BT (k={k})",
                color=SOLVER_STYLE["GD-BT"]["color"])

    _, h, k, st = solve_newton_pure(P.f, P.grad, P.hess, x0,
                                     stopping="relative")
    ax.semilogy([np.linalg.norm(P.grad(xi)) for xi in h],
                label=f"Newton pure (k={k}, spurious -- see S2/S5)",
                color=SOLVER_STYLE["Newton (pure)"]["color"], linestyle="--")

    _, h, k, st, log = solve_newton_damped(P.f, P.grad, P.hess, x0,
                                            stopping="relative")
    ax.semilogy([np.linalg.norm(P.grad(xi)) for xi in h],
                label=f"Newton damped (k={k})",
                color=SOLVER_STYLE["Newton (damped)"]["color"])

    _, h, k, st = solve_momentum(P.f, P.grad, x0,
                                  _f(row["Momentum_alpha"]),
                                  _f(row["Momentum_beta"]),
                                  stopping="relative")
    label = f"Momentum, alpha={_f(row['Momentum_alpha']):.3g} (k={k})"
    if _f(row["Momentum_alpha"]) > 1.0:
        label += " [spurious]"
    ax.semilogy([np.linalg.norm(P.grad(xi)) for xi in h], label=label,
                color=SOLVER_STYLE["Momentum"]["color"],
                linestyle="--" if _f(row["Momentum_alpha"]) > 1.0 else "-")

    _, h, k, st = solve_adam(P.f, P.grad, x0, _f(row["Adam_alpha"]),
                              stopping="relative")
    ax.semilogy([np.linalg.norm(P.grad(xi)) for xi in h],
                label=f"Adam, alpha={_f(row['Adam_alpha']):.3g} (k={k})",
                color=SOLVER_STYLE["Adam"]["color"])

    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$\|\nabla F(z_k)\|$")
    ax.set_title("F3 -- project block (storyline B): convergence\n"
                  "(dashed = converges to a non-minimizer, see S2/S5)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "F3_project_convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    figure_f1()
    figure_f2()
    figure_f3()
