"""Shared tuning protocol (Section 3 of the instructions), used by
run_all.py to fill Table 1 and Table 2.

Grid for alpha: 10^-5 .. 10^0 in half-decades (11 values). Grid for beta
(momentum): {0.5, 0.8, 0.9, 0.95, 0.99}. Report the combination with the
fewest iterations among the runs that reach the stopping rule; runs that
hit the cap or the blow-up guard are "failed". If the best alpha is on
the upper edge of the grid tested so far, extend upward by half-decades
until the best value is interior, stopping at 10^4.
"""
import numpy as np

from optim.gd import solve_gd, solve_gd_bt
from optim.newton import solve_newton_pure, solve_newton_damped
from optim.momentum import solve_momentum
from optim.adam import solve_adam

BASE_EXPONENTS = [round(-5 + 0.5 * i, 1) for i in range(11)]  # -5 .. 0
BETAS = [0.5, 0.8, 0.9, 0.95, 0.99]
MAX_EXPONENT = 4.0


def _alphas_from_exponents(exponents):
    return [10 ** e for e in exponents]


def tune_gd(f, grad, x0, stopping, extra_alphas=None, cap=None):
    """Grid-search alpha for fixed-step GD. Returns a dict with the best
    alpha, its iteration count, whether the grid was extended, and the
    full (alpha -> (k, status)) table actually tried."""
    exponents = list(BASE_EXPONENTS)
    all_results = {}
    extended = False

    def run_batch(exps):
        for e in exps:
            a = 10 ** e
            kwargs = dict(stopping=stopping)
            if cap is not None:
                kwargs["cap"] = cap
            x, h, k, st = solve_gd(f, grad, x0, a, **kwargs)
            all_results[a] = (k, st)

    run_batch(exponents)
    if extra_alphas:
        for a in extra_alphas:
            kwargs = dict(stopping=stopping)
            if cap is not None:
                kwargs["cap"] = cap
            x, h, k, st = solve_gd(f, grad, x0, a, **kwargs)
            all_results[a] = (k, st)

    while True:
        converged = {a: k for a, (k, st) in all_results.items()
                     if st == "converged"}
        if not converged:
            return dict(alpha=None, k=None, status="failed",
                         extended=extended, table=all_results)
        best_a = min(converged, key=converged.get)
        max_exp_tested = max(exponents)
        on_edge = np.isclose(np.log10(best_a), max_exp_tested) and \
            max_exp_tested < MAX_EXPONENT
        if not on_edge:
            return dict(alpha=best_a, k=converged[best_a], status="converged",
                         extended=extended, table=all_results)
        # extend upward by one half-decade
        new_exp = round(max_exp_tested + 0.5, 1)
        exponents.append(new_exp)
        extended = True
        run_batch([new_exp])


def tune_momentum(f, grad, x0, stopping, cap=None):
    exponents = list(BASE_EXPONENTS)
    all_results = {}  # (alpha,beta) -> (k, status)
    extended = False

    def run_batch(exps):
        for e in exps:
            a = 10 ** e
            for b in BETAS:
                kwargs = dict(stopping=stopping)
                if cap is not None:
                    kwargs["cap"] = cap
                x, h, k, st = solve_momentum(f, grad, x0, a, b, **kwargs)
                all_results[(a, b)] = (k, st)

    run_batch(exponents)
    while True:
        converged = {ab: k for ab, (k, st) in all_results.items()
                     if st == "converged"}
        if not converged:
            return dict(alpha=None, beta=None, k=None, status="failed",
                         extended=extended, table=all_results)
        best_ab = min(converged, key=converged.get)
        best_a = best_ab[0]
        max_exp_tested = max(exponents)
        on_edge = np.isclose(np.log10(best_a), max_exp_tested) and \
            max_exp_tested < MAX_EXPONENT
        if not on_edge:
            return dict(alpha=best_ab[0], beta=best_ab[1], k=converged[best_ab],
                         status="converged", extended=extended,
                         table=all_results)
        new_exp = round(max_exp_tested + 0.5, 1)
        exponents.append(new_exp)
        extended = True
        run_batch([new_exp])


def tune_adam(f, grad, x0, stopping, cap=None):
    exponents = list(BASE_EXPONENTS)
    all_results = {}
    extended = False

    def run_batch(exps):
        for e in exps:
            a = 10 ** e
            kwargs = dict(stopping=stopping)
            if cap is not None:
                kwargs["cap"] = cap
            x, h, k, st = solve_adam(f, grad, x0, a, **kwargs)
            all_results[a] = (k, st)

    run_batch(exponents)
    while True:
        converged = {a: k for a, (k, st) in all_results.items()
                     if st == "converged"}
        if not converged:
            return dict(alpha=None, k=None, status="failed",
                         extended=extended, table=all_results)
        best_a = min(converged, key=converged.get)
        max_exp_tested = max(exponents)
        on_edge = np.isclose(np.log10(best_a), max_exp_tested) and \
            max_exp_tested < MAX_EXPONENT
        if not on_edge:
            return dict(alpha=best_a, k=converged[best_a], status="converged",
                         extended=extended, table=all_results)
        new_exp = round(max_exp_tested + 0.5, 1)
        exponents.append(new_exp)
        extended = True
        run_batch([new_exp])


def run_gd_bt(f, grad, x0, stopping, cap=None):
    kwargs = dict(stopping=stopping)
    if cap is not None:
        kwargs["cap"] = cap
    x, hist, k, status, log = solve_gd_bt(f, grad, x0, **kwargs)
    mean_alpha = float(np.mean([a for a, n in log])) if log else float("nan")
    mean_halvings = float(np.mean([n for a, n in log])) if log else float("nan")
    return dict(k=k, status=status, mean_alpha=mean_alpha,
                mean_halvings=mean_halvings, x=x)


def run_newton_pure(f, grad, hess, x0, stopping, cap=None):
    kwargs = dict(stopping=stopping)
    if cap is not None:
        kwargs["cap"] = cap
    x, hist, k, status = solve_newton_pure(f, grad, hess, x0, **kwargs)
    return dict(k=k, status=status, x=x, hist=hist)


def run_newton_damped(f, grad, hess, x0, stopping, cap=None):
    kwargs = dict(stopping=stopping)
    if cap is not None:
        kwargs["cap"] = cap
    x, hist, k, status, log = solve_newton_damped(f, grad, hess, x0, **kwargs)
    n_full_steps = sum(1 for used, a, n in log if n == 0)
    n_grad_fallback = sum(1 for used, a, n in log if not used)
    return dict(k=k, status=status, x=x, hist=hist, log=log,
                n_full_steps=n_full_steps, n_grad_fallback=n_grad_fallback)
