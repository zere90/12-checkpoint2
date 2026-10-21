"""src/bound.py -- M5 (project lead): Task 5, the LP bound and the gap measure.

z_LP <= z* always (the LP relaxation optimizes the same objective over a strictly larger feasible
region: every binary-feasible point is also LP-feasible, since {0,1} subset [0,1]). gap(z, z_lp)
is the relative-gap measure (z - z_lp) / z used throughout Checkpoint 3 and again in Checkpoint 4.
"""
import itertools
import numpy as np


def gap(z, z_lp):
    """(z - z_lp) / z -- how far a feasible value z is above the LP lower bound z_lp, relative to z."""
    return (z - z_lp) / z


def brute_force_binary_optimum(model):
    """Only for tiny models (like the Section 6 triangle problem): try every point in {0,1}^n that
    satisfies every row, return (best_cost, best_point). Used to get the true z* to compare z_LP
    against, and to compute the gap for a few hand-picked feasible points (H5b)."""
    n = model.n_vars
    best_cost, best_point = None, None
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        if _feasible(model, x):
            cost = float(model.c @ x)
            if best_cost is None or cost < best_cost:
                best_cost, best_point = cost, x
    return best_cost, best_point


def _feasible(model, x, tol=1e-9):
    row_val = model.A @ x
    for i, s in enumerate(model.sense):
        if s == '<=' and row_val[i] > model.b[i] + tol:
            return False
        if s == '>=' and row_val[i] < model.b[i] - tol:
            return False
        if s == '=' and abs(row_val[i] - model.b[i]) > tol:
            return False
    if np.any(x < model.lb - tol) or np.any(x > model.ub + tol):
        return False
    return True
