"""src/kkt -- M3 (Momentum-role successor): the checker of optimality conditions (a)-(e), Task 3.

Written as a plain function over arrays (c, A, b, sense, lb, ub, x, pi), not tied to our own
LPModel/LPResult classes, exactly because Task 3 says "the checker... must also accept the
solution of another team's solver": anyone can hand it their own matrices, their own x and their
own pi, as long as pi is reported in each row's own natural sense ('<=' rows get pi<=0, '>=' rows
get pi>=0, '=' rows are free -- the sign convention is checked, not assumed, by condition (b)).

All five conditions follow Section 3 of the Checkpoint 3 handout exactly, with tolerance 1e-6.
"""
from dataclasses import dataclass
import numpy as np

TOL = 1e-6


@dataclass
class KKTReport:
    primal_feasibility: float       # (a) largest violation, rows + bounds
    dual_feasibility: float         # (b) largest sign violation
    complementary_slackness: float  # (c) largest |pi_i * slack_i|
    reduced_costs: float            # (d) largest violation of the reduced-cost conditions
    duality_gap: float              # (e) c^T x - (b^T pi + sum of box-bound terms), signed
    all_satisfied: bool
    tol: float = TOL

    def as_row(self):
        return [self.primal_feasibility, self.dual_feasibility, self.complementary_slackness,
                self.reduced_costs, self.duality_gap]


def check_conditions(c, A, b, sense, lb, ub, x, pi, tol=TOL):
    """c, b, lb, ub, x, pi: 1-D array-like. A: 2-D array-like (m, n). sense: length-m list of
    '<=', '>=', '='. Returns a KKTReport with the five numbers of Table 2."""
    c = np.asarray(c, dtype=float); A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float)
    lb = np.asarray(lb, dtype=float); ub = np.asarray(ub, dtype=float)
    x = np.asarray(x, dtype=float); pi = np.asarray(pi, dtype=float)
    m, n = A.shape

    # (a) primal feasibility: every row and every bound
    row_val = A @ x
    row_violation = np.zeros(m)
    for i in range(m):
        if sense[i] == '<=':
            row_violation[i] = max(0.0, row_val[i] - b[i])
        elif sense[i] == '>=':
            row_violation[i] = max(0.0, b[i] - row_val[i])
        else:  # '='
            row_violation[i] = abs(row_val[i] - b[i])
    bound_violation = np.maximum(0.0, lb - x) + np.maximum(0.0, np.where(np.isinf(ub), -np.inf, x - ub))
    bound_violation = np.nan_to_num(bound_violation, neginf=0.0)
    primal_feasibility = float(max(row_violation.max(initial=0.0), bound_violation.max(initial=0.0)))

    # (b) dual feasibility (signs): pi_i <= 0 for '<=' rows, pi_i >= 0 for '>=' rows, free for '='
    dual_violation = np.zeros(m)
    for i in range(m):
        if sense[i] == '<=':
            dual_violation[i] = max(0.0, pi[i])
        elif sense[i] == '>=':
            dual_violation[i] = max(0.0, -pi[i])
        # '=' : always 0
    dual_feasibility = float(dual_violation.max(initial=0.0))

    # (c) complementary slackness, rows: pi_i * (b_i - a_i^T x) = 0 for every row
    slack = b - row_val
    compl_slackness = float(np.max(np.abs(pi * slack))) if m else 0.0

    # (d) reduced costs: r_j = c_j - A^T pi ; x_j at lb => r_j>=0 ; x_j at ub => r_j<=0 ; interior => r_j=0
    r = c - A.T @ pi
    at_lower = x <= lb + tol
    at_upper = np.isfinite(ub) & (x >= ub - tol)
    rc_violation = np.zeros(n)
    for j in range(n):
        if at_lower[j] and not at_upper[j]:
            rc_violation[j] = max(0.0, -r[j])
        elif at_upper[j] and not at_lower[j]:
            rc_violation[j] = max(0.0, r[j])
        elif not at_lower[j] and not at_upper[j]:
            rc_violation[j] = abs(r[j])
        # fixed variable (lb == ub): no condition
    reduced_costs = float(rc_violation.max(initial=0.0))

    # (e) strong duality: c^T x = b^T pi + sum_j [ lb_j*max(0,r_j) + ub_j*min(0,r_j) if ub_j finite else 0 ]
    # for lb_j = 0 (our variables) this is exactly the handout's "sum of min(0, r_j) over
    # variables with upper bound 1".
    ub_safe = np.where(np.isfinite(ub), ub, 0.0)   # avoid inf * 0 = nan; where() discards this branch anyway
    box_term = lb * np.maximum(0.0, r) + np.where(np.isfinite(ub), ub_safe * np.minimum(0.0, r), 0.0)
    dual_value = float(b @ pi + box_term.sum())
    primal_value = float(c @ x)
    duality_gap = primal_value - dual_value

    all_satisfied = (primal_feasibility <= tol and dual_feasibility <= tol
                      and compl_slackness <= tol and reduced_costs <= tol and abs(duality_gap) <= tol)

    return KKTReport(primal_feasibility, dual_feasibility, compl_slackness, reduced_costs,
                      duality_gap, all_satisfied, tol)


def check_model(model, result, tol=TOL):
    """Convenience wrapper: check_conditions on our own LPModel + LPResult."""
    return check_conditions(model.c, model.A, model.b, model.sense, model.lb, model.ub,
                             result.x, result.pi, tol=tol)
