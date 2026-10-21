"""src/sens -- M4 (Adam-role successor): Task 4, sensitivity of the designated constraint's RHS.

Solves the LP three times (b-delta, b, b+delta), reports the left/right finite-difference slopes
of z(b), and compares with the solver's own dual price pi at b. By convexity of z(.) (the LP value
function), any valid dual price satisfies left slope <= pi <= right slope; if the two slopes
differ, the LP is degenerate at b and pi is not unique there.
"""
from dataclasses import dataclass
import numpy as np
from lp.solve import solve_lp


@dataclass
class SensResult:
    row_index: int
    b0: float
    delta: float
    z_minus: float
    z0: float
    z_plus: float
    left_slope: float
    right_slope: float
    pi: float
    interval_ok: bool   # left_slope <= pi <= right_slope, up to tol
    degenerate: bool     # left_slope != right_slope beyond tol


def rhs_sensitivity(model, row_index, delta=0.01, tol=1e-6):
    b0 = float(model.b[row_index])
    r_minus = solve_lp(model, rhs_override={row_index: b0 - delta})
    r0 = solve_lp(model, rhs_override={row_index: b0})
    r_plus = solve_lp(model, rhs_override={row_index: b0 + delta})

    left_slope = (r0.z - r_minus.z) / delta
    right_slope = (r_plus.z - r0.z) / delta
    pi = float(r0.pi[row_index])

    degenerate = abs(right_slope - left_slope) > tol
    interval_ok = (left_slope - tol) <= pi <= (right_slope + tol)

    return SensResult(row_index, b0, delta, r_minus.z, r0.z, r_plus.z,
                       left_slope, right_slope, pi, interval_ok, degenerate)
