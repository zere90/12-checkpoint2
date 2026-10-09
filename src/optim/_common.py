"""Shared infrastructure: the solver loop, stopping rules, and check_grad.

Owned by the infrastructure role (M5 / shared setup). Every solver in
gd.py, newton.py, momentum.py and adam.py calls `run_loop` with its own
update rule; the loop itself, the stopping rules and the blow-up guard
are identical for every method, exactly as Section 3 of the instructions
requires ("Conventions are fixed so that iteration counts are
comparable").

Interface: every solver returns (x, hist, k, status):
    x      -- final point
    hist   -- list of every iterate, x^0 .. x^k
    k      -- number of UPDATES performed (stopping rule is checked
              before the update, so k counts updates, not checks)
    status -- 'converged' or 'failed' (||x|| > 1e12, or the iteration
              cap was reached: 100000 for gradient methods, 100 for
              Newton)
"""
import numpy as np

CAP_GRAD = 100_000
CAP_NEWTON = 100
BLOWUP = 1e12


def run_loop(grad_fn, x0, update_fn, cap, stopping="absolute", f_fn=None,
             state0=None):
    """Generic optimization loop shared by every solver.

    update_fn(x, g, state) -> (x_new, state_new) performs ONE update.
    stopping: 'absolute'  -> ||grad f(x_k)|| < 1e-6
              'relative'  -> ||grad f(x_k)|| < 1e-6 * ||grad f(x0)||
    """
    x = np.array(x0, dtype=float)
    g0_norm = np.linalg.norm(grad_fn(x0))
    hist = [x.copy()]
    state = state0
    k = 0
    status = "converged"
    while True:
        g = grad_fn(x)
        gn = np.linalg.norm(g)
        if stopping == "absolute":
            done = gn < 1e-6
        elif stopping == "relative":
            done = gn < 1e-6 * g0_norm
        else:
            raise ValueError(f"unknown stopping rule {stopping!r}")
        if done:
            status = "converged"
            break
        if np.linalg.norm(x) > BLOWUP or k >= cap:
            status = "failed"
            break
        x, state = update_fn(x, g, state)
        hist.append(x.copy())
        k += 1
    return x, hist, k, status


def check_grad(f, grad, x, eps=1e-6):
    """Central-difference check of an analytic gradient. Returns the
    max-norm error between the analytic gradient and the central
    difference estimate at x."""
    x = np.asarray(x, dtype=float)
    n = x.size
    numeric = np.zeros(n)
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        numeric[i] = (f(x + e) - f(x - e)) / (2 * eps)
    analytic = np.asarray(grad(x), dtype=float)
    return float(np.max(np.abs(numeric - analytic)))


def check_hess(grad, hess, x, eps=1e-5):
    """Central-difference check of an analytic Hessian against the
    (verified) gradient. Returns the max-norm error."""
    x = np.asarray(x, dtype=float)
    n = x.size
    numeric = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        numeric[:, i] = (grad(x + e) - grad(x - e)) / (2 * eps)
    analytic = np.asarray(hess(x), dtype=float)
    return float(np.max(np.abs(numeric - analytic)))
