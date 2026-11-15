"""Block 3: exact solver (HiGHS via scipy.optimize.milp) on the team's CP3 reference model
(src/model/build.py:build_b2_model), applied to the scaled instance, 120s cap.
"""
import json, sys, time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import LinearConstraint, Bounds

sys.path.insert(0, '.')
import checkpoint4_generator_PUBLIC as gen
from src.model.build import build_b2_model
from experiments.exact_helper import solve_milp

RESULTS = {}
for size in (100, 200, 400):
    d = gen.scaled_instance('B', size, 2)
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
    print(f'size {size}: n_vars={model.n_vars} n_rows={model.n_rows}  solving (120s cap)...', flush=True)
    t0 = time.time()
    res = solve_milp(model.c, constraints, bounds, integrality, limit=120.0)
    wall = time.time() - t0
    res['wall_seconds'] = round(wall, 1)
    RESULTS[size] = res
    print(f'  -> {res}', flush=True)

json.dump(RESULTS, open('results/exact_scaling.json', 'w'), indent=2)
print('wrote results/exact_scaling.json')
