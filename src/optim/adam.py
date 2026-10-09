"""Adam.

Owner: M4 -- Adam role.

    m <- beta1*m + (1-beta1)*g
    s <- beta2*s + (1-beta2)*g^2          (elementwise)
    mhat = m / (1 - beta1^(k+1))
    shat = s / (1 - beta2^(k+1))
    x <- x - alpha * mhat / (sqrt(shat) + eps)
    m^0 = s^0 = 0,  k = 0, 1, 2, ...  counts updates performed so far
"""
import numpy as np

from ._common import run_loop, CAP_GRAD


def solve_adam(f, grad, x0, alpha, beta1=0.9, beta2=0.999, eps=1e-8,
                stopping="absolute", cap=CAP_GRAD):

    def update(x, g, state):
        m, s, k = state
        k = k + 1
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g ** 2
        mhat = m / (1 - beta1 ** k)
        shat = s / (1 - beta2 ** k)
        x = x - alpha * mhat / (np.sqrt(shat) + eps)
        return x, (m, s, k)

    x0a = np.asarray(x0, dtype=float)
    state0 = (np.zeros_like(x0a), np.zeros_like(x0a), 0)
    return run_loop(grad, x0, update, cap, stopping=stopping, f_fn=f,
                     state0=state0)
