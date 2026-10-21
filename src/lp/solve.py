"""src/lp -- M2 (Newton-role successor): solving the LP relaxation and reading its solution.

Any LP solver is allowed (Section 4); we use scipy.optimize.linprog(method="highs"), which ships
with scipy and needs no separate install (PuLP<4 + CBC gives the same z_LP to 1e-6 -- we checked).

Rows arrive in their own natural sense ('<=','>=','=') from src/model. scipy's linprog wants
A_eq/b_eq for '=' rows and A_ub/b_ub (only '<=') for inequalities, so every '>=' row is negated
before solving. Section 2's hint for H1(b) is exactly this: negating a row to flip '>=' to '<='
flips the sign of its dual price. We undo that flip on the way out, so the duals this module
returns (LPResult.pi) are always in the row's own original sense -- the convention src/kkt and
src/sens both assume, and the one Table 2's sign-convention check (condition (b)) is stated in.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import linprog


@dataclass
class LPResult:
    x: np.ndarray            # (n,) primal solution
    z: float                 # objective value
    pi: np.ndarray           # (m,) dual prices, one per row, in the row's OWN natural sense
    reduced_cost: np.ndarray  # (n,) r_j = c_j - sum_i pi_i * A_ij
    solver: str
    time_sec: float
    status: str
    n_fractional: int        # number of 'y' variables strictly inside (0, 1)
    fractional_vars: list    # [(label, value), ...]


def solve_lp(model, rhs_override=None):
    """Solve model's LP relaxation. rhs_override: optional dict {row_index: new_b_i} used by
    src/sens for the +-0.01 perturbation of Task 4, without having to rebuild the whole model."""
    A, b, sense = model.A, model.b.copy(), model.sense
    if rhs_override:
        for i, bi in rhs_override.items():
            b[i] = bi

    m = model.n_rows
    eq_rows = [i for i in range(m) if sense[i] == '=']
    le_rows = [i for i in range(m) if sense[i] == '<=']
    ge_rows = [i for i in range(m) if sense[i] == '>=']

    A_eq = A[eq_rows, :] if eq_rows else None
    b_eq = b[eq_rows] if eq_rows else None

    # '<=' rows go through unchanged; '>=' rows are negated to become '<='
    ub_row_order = le_rows + ge_rows
    if ub_row_order:
        A_ub_parts = [A[le_rows, :]] if le_rows else []
        b_ub_parts = [b[le_rows]] if le_rows else []
        if ge_rows:
            A_ub_parts.append(-A[ge_rows, :])
            b_ub_parts.append(-b[ge_rows])
        A_ub = np.vstack(A_ub_parts)
        b_ub = np.concatenate(b_ub_parts)
    else:
        A_ub = b_ub = None

    bounds = list(zip(model.lb, [None if np.isinf(u) else u for u in model.ub]))

    t0 = time.perf_counter()
    res = linprog(model.c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    elapsed = time.perf_counter() - t0

    if not res.success:
        raise RuntimeError(f"LP did not solve: {res.message}")

    # rebuild the natural-sense dual vector pi, length m, in original row order
    pi = np.zeros(m)
    if eq_rows:
        pi_eq = np.atleast_1d(res.eqlin.marginals)
        for idx, row_i in enumerate(eq_rows):
            pi[row_i] = pi_eq[idx]
    if ub_row_order:
        pi_ub = np.atleast_1d(res.ineqlin.marginals)
        for idx, row_i in enumerate(ub_row_order):
            sign = 1.0 if row_i in le_rows else -1.0   # undo the '>=' -> '<=' negation
            pi[row_i] = sign * pi_ub[idx]

    reduced_cost = model.c - model.A.T @ pi

    x = res.x
    fractional = []
    for j in range(model.n_vars):
        if model.var_kind[j] == 'y' and model.ub[j] > 0 and 1e-9 < x[j] < 1 - 1e-9:
            fractional.append((model.var_label[j], float(x[j])))

    return LPResult(
        x=x, z=float(res.fun), pi=pi, reduced_cost=reduced_cost,
        solver='scipy.optimize.linprog(method="highs")', time_sec=elapsed, status=res.message,
        n_fractional=len(fractional), fractional_vars=fractional,
    )
