"""Newton's method: pure and damped (Armijo-backtracked, descent-safeguarded).

Owner: M2 -- Newton role.

    Newton (pure)    solve H p = -grad f  (np.linalg.solve);  x <- x + p
    Newton (damped)  p as above; if <grad f, p> >= 0 use p = -grad f;
                      then backtrack with d = p (alpha0 = 1, c = 1e-4)
"""
import numpy as np

from ._common import run_loop, CAP_NEWTON


def solve_newton_pure(f, grad, hess, x0, stopping="absolute",
                       cap=CAP_NEWTON):

    def update(x, g, state):
        H = hess(x)
        p = np.linalg.solve(H, -g)
        return x + p, state

    return run_loop(grad, x0, update, cap, stopping=stopping, f_fn=f)


def solve_newton_damped(f, grad, hess, x0, stopping="absolute",
                         cap=CAP_NEWTON, c=1e-4, alpha0=1.0):
    log = []  # (used_newton_direction, alpha, n_halvings) per step

    def update(x, g, state):
        H = hess(x)
        p = np.linalg.solve(H, -g)
        used_newton = True
        if g @ p >= 0:
            p = -g
            used_newton = False
        alpha = alpha0
        fx = f(x)
        dot = g @ p
        n_halve = 0
        while f(x + alpha * p) > fx + c * alpha * dot:
            alpha /= 2.0
            n_halve += 1
        log.append((used_newton, alpha, n_halve))
        return x + alpha * p, state

    x, hist, k, status = run_loop(grad, x0, update, cap, stopping=stopping,
                                   f_fn=f)
    return x, hist, k, status, log
