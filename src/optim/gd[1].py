"""Gradient descent: fixed step size and backtracking (Armijo) line search.

Owner: M1 -- GD role.

    GD     x <- x - alpha * grad f(x)
    GD-BT  d = -grad f(x); start alpha = 1;
           while f(x + alpha*d) > f(x) + 1e-4 * alpha * <grad f(x), d>:
               alpha <- alpha / 2
           x <- x + alpha*d
"""
import numpy as np

from ._common import run_loop, CAP_GRAD


def solve_gd(f, grad, x0, alpha, stopping="absolute", cap=CAP_GRAD):
    """Fixed-step gradient descent."""

    def update(x, g, state):
        return x - alpha * g, state

    return run_loop(grad, x0, update, cap, stopping=stopping, f_fn=f)


def solve_gd_bt(f, grad, x0, stopping="absolute", cap=CAP_GRAD, c=1e-4,
                 alpha0=1.0):
    """Gradient descent with Armijo backtracking line search."""
    halvings_log = []

    def update(x, g, state):
        d = -g
        alpha = alpha0
        fx = f(x)
        dot = g @ d
        n_halve = 0
        while f(x + alpha * d) > fx + c * alpha * dot:
            alpha /= 2.0
            n_halve += 1
        halvings_log.append((alpha, n_halve))
        return x + alpha * d, state

    x, hist, k, status = run_loop(grad, x0, update, cap, stopping=stopping,
                                   f_fn=f)
    return x, hist, k, status, halvings_log
