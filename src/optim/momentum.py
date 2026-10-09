"""Classical (heavy-ball) momentum.

Owner: M3 -- Momentum role.

    v <- beta*v + grad f(x);   x <- x - alpha*v;   v^0 = 0
"""
import numpy as np

from ._common import run_loop, CAP_GRAD


def solve_momentum(f, grad, x0, alpha, beta, stopping="absolute",
                    cap=CAP_GRAD):

    def update(x, g, state):
        v = state
        v = beta * v + g
        return x - alpha * v, v

    v0 = np.zeros_like(np.asarray(x0, dtype=float))
    return run_loop(grad, x0, update, cap, stopping=stopping, f_fn=f,
                     state0=v0)
