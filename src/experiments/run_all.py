"""python -m experiments.run_all

Regenerates every number behind Table 1, Table 2 and the S1-S4 analysis
sections, and writes them to results/*.csv. Fixed starts, no randomness.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from problems import rosenbrock as Rb
from problems import quadratic as Q
from problems import project as P
from optim.gd import solve_gd, solve_gd_bt
from optim.newton import solve_newton_pure, solve_newton_damped
from optim.momentum import solve_momentum
from optim.adam import solve_adam
from optim._common import check_grad, check_hess
from experiments.tuning import (tune_gd, tune_momentum, tune_adam,
                                 run_gd_bt, run_newton_pure, run_newton_damped,
                                 BETAS)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def fmt(k, status):
    return str(k) if status == "converged" else "---"


# --------------------------------------------------------------------
# 0. check_grad / check_hess on every problem (must all be ~1e-6 or less)
# --------------------------------------------------------------------
def section0_checks():
    rows = []
    rows.append(["problem", "check_grad", "check_hess"])
    rows.append(["R1", check_grad(Rb.f, Rb.grad, Rb.R1_START),
                 check_hess(Rb.grad, Rb.hess, Rb.R1_START)])
    rows.append(["R2", check_grad(Rb.f, Rb.grad, Rb.R2_START),
                 check_hess(Rb.grad, Rb.hess, Rb.R2_START)])
    rows.append(["Q1", check_grad(Q.q1_f, Q.q1_grad, Q.Q1_START),
                 check_hess(Q.q1_grad, Q.q1_hess, Q.Q1_START)])
    rows.append(["Q2", check_grad(Q.q2_f, Q.q2_grad, Q.Q2_START),
                 check_hess(Q.q2_grad, Q.q2_hess, Q.Q2_START)])
    rows.append(["Project", check_grad(P.f, P.grad, P.START), "n/a (central-diff Hessian)"])
    with open(os.path.join(RESULTS, "checks.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print("check_grad/check_hess:")
    for r in rows:
        print(" ", r)
    return rows


# --------------------------------------------------------------------
# Table 1: test-bed (R1, R2, Q1, Q2) x 6 solvers
# --------------------------------------------------------------------
def solve_problem_row(name, f, grad, hess, x0, stopping, extra_gd_alphas=None):
    row = {"problem": name}

    gd_res = tune_gd(f, grad, x0, stopping, extra_alphas=extra_gd_alphas)
    row["GD_alpha"] = gd_res["alpha"]
    row["GD_k"] = fmt(gd_res["k"], gd_res["status"])
    row["GD_extended"] = gd_res["extended"]

    bt_res = run_gd_bt(f, grad, x0, stopping)
    row["GDBT_k"] = fmt(bt_res["k"], bt_res["status"])
    row["GDBT_mean_alpha"] = bt_res["mean_alpha"]
    row["GDBT_mean_halvings"] = bt_res["mean_halvings"]

    np_res = run_newton_pure(f, grad, hess, x0, stopping)
    row["NewtonPure_k"] = fmt(np_res["k"], np_res["status"])

    nd_res = run_newton_damped(f, grad, hess, x0, stopping)
    row["NewtonDamped_k"] = fmt(nd_res["k"], nd_res["status"])

    mom_res = tune_momentum(f, grad, x0, stopping)
    row["Momentum_alpha"] = mom_res["alpha"]
    row["Momentum_beta"] = mom_res["beta"]
    row["Momentum_k"] = fmt(mom_res["k"], mom_res["status"])
    row["Momentum_extended"] = mom_res["extended"]

    adam_res = tune_adam(f, grad, x0, stopping)
    row["Adam_alpha"] = adam_res["alpha"]
    row["Adam_k"] = fmt(adam_res["k"], adam_res["status"])
    row["Adam_extended"] = adam_res["extended"]

    return row


def table1():
    rows = []
    print("Table 1: R1 ...")
    rows.append(solve_problem_row("R1", Rb.f, Rb.grad, Rb.hess, Rb.R1_START,
                                   "absolute"))
    print("Table 1: R2 ...")
    rows.append(solve_problem_row("R2", Rb.f, Rb.grad, Rb.hess, Rb.R2_START,
                                   "absolute"))
    print("Table 1: Q1 ...")
    alpha_star_q = 1.0 / (1.0 + Q.C)
    rows.append(solve_problem_row("Q1", Q.q1_f, Q.q1_grad, Q.q1_hess,
                                   Q.Q1_START, "relative",
                                   extra_gd_alphas=[alpha_star_q]))
    print("Table 1: Q2 ...")
    rows.append(solve_problem_row("Q2", Q.q2_f, Q.q2_grad, Q.q2_hess,
                                   Q.Q2_START, "relative",
                                   extra_gd_alphas=[alpha_star_q]))

    fields = list(rows[0].keys())
    with open(os.path.join(RESULTS, "table1.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("Table 1 done.")
    return rows


def table2():
    row = solve_problem_row("Project (storyline B)", P.f, P.grad, P.hess,
                             P.START, "relative")
    fields = list(row.keys())
    with open(os.path.join(RESULTS, "table2.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(row)
    print("Table 2 done.")
    return row


# --------------------------------------------------------------------
# S1: GD stability on Q1 (t * 2/lambda_max); GD-BT mean alpha/halvings
# --------------------------------------------------------------------
def section_s1():
    rows = []
    lam_max = Q.LAMBDA_MAX
    eigs = np.array([Q.LAMBDA_MIN, Q.LAMBDA_MAX])
    for t in [0.5, 0.9, 0.99, 1.01, 1.1]:
        a = t * 2.0 / lam_max
        x, h, k, st = solve_gd(Q.q1_f, Q.q1_grad, Q.Q1_START, a,
                                stopping="relative")
        rho_pred = max(abs(1 - a * li) for li in eigs)
        # measured decay: geometric mean of ||g_{i+1}||/||g_i|| over the run
        if st == "converged" and len(h) > 2:
            gnorms = [np.linalg.norm(Q.q1_grad(xi)) for xi in h]
            ratios = [gnorms[i + 1] / gnorms[i] for i in range(len(gnorms) - 1)
                      if gnorms[i] > 0]
            measured = float(np.exp(np.mean(np.log(ratios)))) if ratios else float("nan")
        else:
            measured = float("nan")
        rows.append(dict(t=t, alpha=a, k=fmt(k, st), status=st,
                          rho_predicted=rho_pred, rho_measured=measured))
    with open(os.path.join(RESULTS, "s1_gd_stability.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    bt_rows = []
    for name, f, grad, x0, stopping in [
        ("R1", Rb.f, Rb.grad, Rb.R1_START, "absolute"),
        ("Q1", Q.q1_f, Q.q1_grad, Q.Q1_START, "relative"),
        ("Project", P.f, P.grad, P.START, "relative"),
    ]:
        res = run_gd_bt(f, grad, x0, stopping)
        bt_rows.append(dict(problem=name, k=fmt(res["k"], res["status"]),
                             mean_alpha=res["mean_alpha"],
                             mean_halvings=res["mean_halvings"]))
    with open(os.path.join(RESULTS, "s1_gdbt_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bt_rows[0].keys()))
        w.writeheader()
        w.writerows(bt_rows)
    print("S1 done.")
    return rows, bt_rows


# --------------------------------------------------------------------
# S2: iterations vs kappa (c in {10,100,1000,277}); pure vs damped Newton
# --------------------------------------------------------------------
def section_s2(table1_rows=None):
    cached_mom_adam = {}
    if table1_rows:
        for r in table1_rows:
            if r["problem"] == "Q1":
                cached_mom_adam[Q.C] = r
    rows = []
    for c in [10, 100, 1000, Q.C]:
        def f(x, c=c): return x[0] ** 2 + c * x[1] ** 2
        def g(x, c=c): return np.array([2 * x[0], 2 * c * x[1]])
        def hh(x, c=c): return np.array([[2.0, 0.0], [0.0, 2.0 * c]])
        x0 = Q.X0_QUAD
        alpha_star = 1.0 / (1.0 + c)
        xg, hg, kg, stg = solve_gd(f, g, x0, alpha_star, stopping="relative")
        xn, hn, kn, stn = solve_newton_pure(f, g, hh, x0, stopping="relative")
        if c in cached_mom_adam:
            r = cached_mom_adam[c]
            mom = dict(k=r["Momentum_k"], status="converged" if r["Momentum_k"] != "---" else "failed",
                       alpha=r["Momentum_alpha"], beta=r["Momentum_beta"])
            ad = dict(k=r["Adam_k"], status="converged" if r["Adam_k"] != "---" else "failed",
                      alpha=r["Adam_alpha"])
            mom_k_str, ad_k_str = r["Momentum_k"], r["Adam_k"]
        else:
            mom = tune_momentum(f, g, x0, "relative")
            ad = tune_adam(f, g, x0, "relative")
            mom_k_str, ad_k_str = fmt(mom["k"], mom["status"]), fmt(ad["k"], ad["status"])
        rows.append(dict(c=c, GD_alpha_star_k=fmt(kg, stg),
                          Momentum_k=mom_k_str,
                          Momentum_alpha=mom["alpha"], Momentum_beta=mom["beta"],
                          Adam_k=ad_k_str, Adam_alpha=ad["alpha"],
                          Newton_k=fmt(kn, stn)))
    with open(os.path.join(RESULTS, "s2_kappa.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pd_rows = []
    for name, f, grad, hess, x0, stopping in [
        ("R2", Rb.f, Rb.grad, Rb.hess, Rb.R2_START, "absolute"),
        ("Project", P.f, P.grad, P.hess, P.START, "relative"),
    ]:
        pure = run_newton_pure(f, grad, hess, x0, stopping)
        damp = run_newton_damped(f, grad, hess, x0, stopping)
        x_pure = pure["x"]
        x_damp = damp["x"]
        eig_pure = np.linalg.eigvalsh(hess(x_pure))
        eig_damp = np.linalg.eigvalsh(hess(x_damp))
        f_hist_pure = [f(xi) for xi in pure["hist"]]
        f_hist_damp = [f(xi) for xi in damp["hist"]]
        mono_pure = all(f_hist_pure[i + 1] <= f_hist_pure[i] + 1e-12
                         for i in range(len(f_hist_pure) - 1))
        mono_damp = all(f_hist_damp[i + 1] <= f_hist_damp[i] + 1e-12
                         for i in range(len(f_hist_damp) - 1))
        pd_rows.append(dict(problem=name,
                             pure_k=fmt(pure["k"], pure["status"]),
                             pure_monotone=mono_pure,
                             pure_min_eig=float(np.min(eig_pure)),
                             damped_k=fmt(damp["k"], damp["status"]),
                             damped_monotone=mono_damp,
                             damped_min_eig=float(np.min(eig_damp)),
                             damped_n_grad_fallback=damp["n_grad_fallback"]))
    with open(os.path.join(RESULTS, "s2_pure_vs_damped.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pd_rows[0].keys()))
        w.writeheader()
        w.writerows(pd_rows)
    print("S2 done.")
    return rows, pd_rows


# --------------------------------------------------------------------
# S3: momentum beta sweep on R1 (alpha=1e-3); tuned vs Polyak heavy-ball on Q1
# --------------------------------------------------------------------
def section_s3():
    rows = []
    for b in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]:
        x, h, k, st = solve_momentum(Rb.f, Rb.grad, Rb.R1_START, 1e-3, b,
                                      stopping="absolute")
        fs = [Rb.f(xi) for xi in h]
        ever_increased = any(fs[i + 1] > fs[i] + 1e-12 for i in range(len(fs) - 1))
        rows.append(dict(beta=b, k=fmt(k, st), status=st,
                          f_ever_increased=ever_increased))
    with open(os.path.join(RESULTS, "s3_beta_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lam_min, lam_max, kappa = Q.LAMBDA_MIN, Q.LAMBDA_MAX, Q.KAPPA
    alpha_hb = 4.0 / (np.sqrt(lam_max) + np.sqrt(lam_min)) ** 2
    beta_hb = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2
    x, h, k_hb, st_hb = solve_momentum(Q.q1_f, Q.q1_grad, Q.Q1_START,
                                        alpha_hb, beta_hb, stopping="relative")
    tuned = tune_momentum(Q.q1_f, Q.q1_grad, Q.Q1_START, "relative")
    alpha_star = 1.0 / (1.0 + Q.C)
    xg, hg, kg, stg = solve_gd(Q.q1_f, Q.q1_grad, Q.Q1_START, alpha_star,
                                stopping="relative")
    with open(os.path.join(RESULTS, "s3_heavyball.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "alpha", "beta", "k", "status"])
        w.writerow(["heavy-ball (Polyak)", alpha_hb, beta_hb, fmt(k_hb, st_hb), st_hb])
        w.writerow(["tuned momentum", tuned["alpha"], tuned["beta"],
                    fmt(tuned["k"], tuned["status"]), tuned["status"]])
        w.writerow(["plain GD, alpha*", alpha_star, "-", fmt(kg, stg), stg])
        w.writerow(["sqrt(kappa)", "-", "-", np.sqrt(kappa), "-"])
    print("S3 done.")
    return rows


# --------------------------------------------------------------------
# S4: Adam sensitivity to alpha on R1, Q1, Project
# --------------------------------------------------------------------
def section_s4():
    rows = []
    for name, f, grad, x0, stopping in [
        ("R1", Rb.f, Rb.grad, Rb.R1_START, "absolute"),
        ("Q1", Q.q1_f, Q.q1_grad, Q.Q1_START, "relative"),
        ("Project", P.f, P.grad, P.START, "relative"),
    ]:
        for e in BASE_EXP_FOR_S4:
            a = 10 ** e
            x, h, k, st = solve_adam(f, grad, x0, a, stopping=stopping)
            rows.append(dict(problem=name, alpha=a, k=fmt(k, st), status=st))
    with open(os.path.join(RESULTS, "s4_adam_alpha.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("S4 done.")
    return rows


BASE_EXP_FOR_S4 = [-5, -4, -3, -2, -1, 0]  # includes the Adam default 1e-3


if __name__ == "__main__":
    section0_checks()
    t1_rows = table1()
    table2()
    section_s1()
    section_s2(table1_rows=t1_rows)
    section_s3()
    section_s4()
    print("All results written to", os.path.abspath(RESULTS))
