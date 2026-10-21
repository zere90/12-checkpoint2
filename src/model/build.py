 """src/model -- M1 (GD-role successor): data loading and matrix-form bookkeeping.

Builds the reference model of the Checkpoint 3 handout (Section 2, storyline B, variant 2 --
cloud resource allocation with zones) in a solver-agnostic matrix form, and the tiny "triangle
problem" of Section 6 used by every role's hand task.

Design choice: rows are kept in their OWN natural sense ('<=', '>=', '=') rather than forced into
scipy's all-'<=' convention. This is what Task 3 (H1b) asks you to reason about by hand -- the
dual price of a row changes sign when you flip its sense by multiplying by -1 -- so the LP solver
(src/lp) does that flip internally and flips the reported dual back, and the KKT checker (src/kkt)
always works in the natural sense. That keeps the checker usable on anyone's solver output, as
Task 3 requires, as long as they report duals in the same natural-sense convention.
"""
from dataclasses import dataclass, field
import numpy as np

# tangent points of g(u) = u / (1 - u), fixed for everyone (Section 2, storyline B)
TANGENT_POINTS = (0.0, 0.4, 0.7, 0.9)


@dataclass
class LPModel:
    c: np.ndarray                  # (n,) objective coefficients
    A: np.ndarray                  # (m, n) constraint matrix, rows in their natural sense
    b: np.ndarray                  # (m,) right-hand sides
    sense: list                    # length m, each one of '<=', '>=', '='
    lb: np.ndarray                 # (n,) lower bounds
    ub: np.ndarray                 # (n,) upper bounds (np.inf = unbounded)
    row_family: list               # length m, e.g. 'B1', 'B2', 'B3', 'B4', 'T1'...
    row_label: list                # length m, human-readable label
    var_label: list                # length n, human-readable label
    var_kind: list                 # length n, e.g. 'y' or 't'
    meta: dict = field(default_factory=dict)

    @property
    def n_vars(self):
        return self.c.shape[0]

    @property
    def n_rows(self):
        return self.A.shape[0]


def _b4_coeffs():
    """a_m, b_m for m = 1..4 from Section 2: a_m = 1/(1-u_m)^2, b_m = -u_m^2/(1-u_m)^2."""
    a, b = [], []
    for u in TANGENT_POINTS:
        denom = (1 - u) ** 2
        a.append(1.0 / denom)
        b.append(-(u ** 2) / denom)
    return a, b


def build_b2_model(d):
    """Build the reference model (Section 2, storyline B, variant 2 -- zones) for instance d
    (the dict returned by cp34_instance(seed, 'B', 2)).

    Variables, in column order: y[k,l] for k=0..M-1, l=0..S-1 (row-major in k, i.e. index
    k*S+l), then t[l] for l=0..S-1. Variables of forbidden (zone-mismatched) VM/server pairs are
    KEPT, with upper bound 0, per the handout's check-value note -- not deleted -- so that the
    variable count matches the task file's check value (360 here).

    Rows, in order: (B1) one row per VM k (=), (B2) one row per server l (<=), (B3) one row per
    server l (<=), (B4) one row per (server l, tangent m) pair (>=).
    """
    S, M = d['S'], d['M']
    cpu_cap, mem_cap = d['cpu_cap'], d['mem_cap']
    cpu_req, mem_req = d['cpu_req'], d['mem_req']
    zone, vm_required_zone = d['zone'], d['vm_required_zone']

    n_y = M * S
    n_t = S
    n = n_y + n_t

    def idx_y(k, l):
        return k * S + l

    def idx_t(l):
        return n_y + l

    # objective: minimize sum_l t_l
    c = np.zeros(n)
    for l in range(S):
        c[idx_t(l)] = 1.0

    # bounds
    lb = np.zeros(n)
    ub = np.ones(n)
    ub[n_y:] = np.inf   # t_l has no upper bound
    n_forbidden = 0
    for k in range(M):
        req = vm_required_zone[k]
        if req == -1:
            continue
        for l in range(S):
            if zone[l] != req:
                ub[idx_y(k, l)] = 0.0    # forbidden pair: keep the variable, fix it to 0
                n_forbidden += 1

    rows_A, rows_b, rows_sense, rows_family, rows_label = [], [], [], [], []

    # (B1) sum_l y_kl = 1, for every VM k
    for k in range(M):
        row = np.zeros(n)
        for l in range(S):
            row[idx_y(k, l)] = 1.0
        rows_A.append(row); rows_b.append(1.0); rows_sense.append('='); rows_family.append('B1')
        rows_label.append(f"B1: VM {k} placed exactly once")

    # (B2) sum_k cpu_req[k] y_kl <= cpu_cap[l], for every server l
    for l in range(S):
        row = np.zeros(n)
        for k in range(M):
            row[idx_y(k, l)] = cpu_req[k]
        rows_A.append(row); rows_b.append(float(cpu_cap[l])); rows_sense.append('<='); rows_family.append('B2')
        rows_label.append(f"B2: server {l} CPU capacity")

    # (B3) sum_k mem_req[k] y_kl <= mem_cap[l], for every server l
    for l in range(S):
        row = np.zeros(n)
        for k in range(M):
            row[idx_y(k, l)] = mem_req[k]
        rows_A.append(row); rows_b.append(float(mem_cap[l])); rows_sense.append('<='); rows_family.append('B3')
        rows_label.append(f"B3: server {l} memory capacity")

    # (B4) t_l >= a_m * (sum_k cpu_req[k] y_kl) / C_l + b_m, for every l and m=1..4
    a_coef, b_coef = _b4_coeffs()
    for l in range(S):
        C_l = float(cpu_cap[l])
        for m in range(4):
            row = np.zeros(n)
            row[idx_t(l)] = 1.0
            for k in range(M):
                row[idx_y(k, l)] = -a_coef[m] * cpu_req[k] / C_l
            rows_A.append(row); rows_b.append(b_coef[m]); rows_sense.append('>='); rows_family.append('B4')
            rows_label.append(f"B4: server {l}, tangent m={m+1} (u={TANGENT_POINTS[m]})")

    var_label = [f"y[{k},{l}]" for k in range(M) for l in range(S)] + [f"t[{l}]" for l in range(S)]
    var_kind = ['y'] * n_y + ['t'] * n_t

    model = LPModel(
        c=c, A=np.array(rows_A), b=np.array(rows_b), sense=rows_sense,
        lb=lb, ub=ub, row_family=rows_family, row_label=rows_label,
        var_label=var_label, var_kind=var_kind,
        meta=dict(S=S, M=M, n_y=n_y, n_t=n_t, n_forbidden=n_forbidden,
                   idx_y=idx_y, idx_t=idx_t, cpu_cap=cpu_cap, mem_cap=mem_cap,
                   cpu_req=cpu_req, mem_req=mem_req),
    )
    return model


def build_triangle_model(c1, c2, c3):
    """The Section 6 triangle problem: minimize c1*y1+c2*y2+c3*y3 s.t.
    (T1) y1+y2>=1, (T2) y2+y3>=1, (T3) y1+y3>=1, 0<=y<=1. Used by every role's hand-task test."""
    c = np.array([c1, c2, c3], dtype=float)
    A = np.array([
        [1.0, 1.0, 0.0],   # T1
        [0.0, 1.0, 1.0],   # T2
        [1.0, 0.0, 1.0],   # T3
    ])
    b = np.array([1.0, 1.0, 1.0])
    sense = ['>=', '>=', '>=']
    lb = np.zeros(3)
    ub = np.ones(3)
    return LPModel(
        c=c, A=A, b=b, sense=sense, lb=lb, ub=ub,
        row_family=['T1', 'T2', 'T3'], row_label=['T1: y1+y2>=1', 'T2: y2+y3>=1', 'T3: y1+y3>=1'],
        var_label=['y1', 'y2', 'y3'], var_kind=['y', 'y', 'y'],
        meta=dict(triangle=True),
    )
