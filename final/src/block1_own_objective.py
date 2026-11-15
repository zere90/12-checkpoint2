"""Block 1: evaluate the team's own Checkpoint-1 objective F(x) = sum_j n_j(x) * u_j(x)^2
on (a) the reference model's proven-optimal assignment and (b) the best approximate-method
(SA) assignment, both for the CP1/CP3/CP4 instance seed=12, variant=2 (storyline B).
"""
import json
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import LinearConstraint, Bounds, milp

sys.path.insert(0, '.')
import checkpoint4_generator_PUBLIC as gen
from src.model.build import build_b2_model
from src.meta.sa import solve as sa_solve

d = gen.cp34_instance(12, 'B', 2)
S, M = d['S'], d['M']
cpu_req, cpu_cap = d['cpu_req'], d['cpu_cap']


def cp1_objective(assign):
    """CP1's own F(x) = sum_j n_j(x) * u_j(x)^2, assign[k] = server index of VM k."""
    total = 0.0
    for j in range(S):
        vms_on_j = [k for k in range(M) if assign[k] == j]
        nj = len(vms_on_j)
        if nj == 0:
            continue
        uj = sum(cpu_req[k] for k in vms_on_j) / cpu_cap[j]
        total += nj * uj ** 2
    return total


def reference_objective(assign):
    """Reference model's own latency, sum_l t_l using the piecewise-linear lower envelope,
    for a sanity check against the exact/SA-reported value."""
    from src.meta.sa import latency
    total = 0.0
    for j in range(S):
        vms_on_j = [k for k in range(M) if assign[k] == j]
        if not vms_on_j:
            continue
        uj = sum(cpu_req[k] for k in vms_on_j) / cpu_cap[j]
        total += latency(uj)
    return total


# --- (a) exact optimum of the reference model ---
model = build_b2_model(d)
A = model.A
lo = np.full(model.n_rows, -np.inf)
hi = np.full(model.n_rows, np.inf)
for i, s in enumerate(model.sense):
    if s == '=':
        lo[i] = hi[i] = model.b[i]
    elif s == '<=':
        hi[i] = model.b[i]
    elif s == '>=':
        lo[i] = model.b[i]
constraints = LinearConstraint(csr_matrix(A), lo, hi)
bounds = Bounds(model.lb, [1e9 if np.isinf(u) else u for u in model.ub])
integrality = np.array([1 if k == 'y' else 0 for k in model.var_kind])

res = milp(model.c, constraints=constraints, integrality=integrality, bounds=bounds,
           options=dict(time_limit=60, mip_rel_gap=1e-9, disp=False))
x = res.x
n_y = M * S
y = x[:n_y].reshape(M, S)
assign_exact = [int(np.argmax(y[k])) for k in range(M)]
ref_obj_exact = float(res.fun)
cp1_obj_exact = cp1_objective(assign_exact)
ref_check_exact = reference_objective(assign_exact)

# --- (b) SA, seed 0 (ties the proven optimum per Table 2, std=0 across all 10 seeds) ---
sa_res = sa_solve(d, 2, 0, 100000, params={"T0": 0.820689, "Tend": 0.000795}, start=None)
assign_sa = sa_res['solution']
ref_obj_sa = sa_res['cost']
cp1_obj_sa = cp1_objective(assign_sa)
ref_check_sa = reference_objective(assign_sa)
same_assignment = list(assign_exact) == list(assign_sa)

out = dict(
    ref_obj_exact=ref_obj_exact, ref_check_exact=ref_check_exact, cp1_obj_exact=cp1_obj_exact,
    ref_obj_sa=ref_obj_sa, ref_check_sa=ref_check_sa, cp1_obj_sa=cp1_obj_sa,
    same_assignment=same_assignment,
)
print(json.dumps(out, indent=2))
json.dump(out, open('results/block1_own_objective.json', 'w'), indent=2)
