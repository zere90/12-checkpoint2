"""Block 1: evaluate the team's own Checkpoint-1 objective F(x) = sum_j n_j(x) * u_j(x)^2
on (a) the reference model's proven-optimal assignment, (b) all ten SA seeds (all of which tie
z_ref exactly per Table 2), and (c) the exact solver's assignment, for the CP1/CP3/CP4 instance
seed=12, variant=2 (storyline B). Extended per Tilesbay's review (FINAL_SIGNOFF.md, Block 1):
report the full range across all ten tied-optimal SA seeds, not just seed 0.
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
    from src.meta.sa import latency
    total = 0.0
    for j in range(S):
        vms_on_j = [k for k in range(M) if assign[k] == j]
        if not vms_on_j:
            continue
        uj = sum(cpu_req[k] for k in vms_on_j) / cpu_cap[j]
        total += latency(uj)
    return total


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

SA_PARAMS = {"T0": 0.820689, "Tend": 0.000795}
sa_per_seed = []
for seed in range(10):
    sa_res = sa_solve(d, 2, seed, 100000, params=SA_PARAMS, start=None)
    assign_sa = sa_res['solution']
    sa_per_seed.append(dict(
        seed=seed,
        ref_obj=sa_res['cost'],
        ref_check=reference_objective(assign_sa),
        cp1_obj=cp1_objective(assign_sa),
        same_as_exact=(list(assign_sa) == list(assign_exact)),
    ))

cp1_vals = [s['cp1_obj'] for s in sa_per_seed]
seed0 = sa_per_seed[0]

out = dict(
    params=SA_PARAMS,
    ref_obj_exact=ref_obj_exact, ref_check_exact=ref_check_exact, cp1_obj_exact=cp1_obj_exact,
    ref_obj_sa=seed0['ref_obj'], ref_check_sa=seed0['ref_check'], cp1_obj_sa=seed0['cp1_obj'],
    same_assignment=seed0['same_as_exact'],
    sa_all_seeds=sa_per_seed,
    cp1_obj_sa_min=min(cp1_vals), cp1_obj_sa_max=max(cp1_vals),
)
print(json.dumps(out, indent=2))
json.dump(out, open('results/final_block1_own_objective.json', 'w'), indent=2)
