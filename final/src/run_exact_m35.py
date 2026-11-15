import json, time, sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import LinearConstraint, Bounds, milp
sys.path.insert(0, '.')
import checkpoint4_generator_PUBLIC as gen
from src.model.build import build_b2_model

d = gen.cp34_instance(12, 'B', 2)
model = build_b2_model(d)
A = model.A
lo = np.full(model.n_rows, -np.inf); hi = np.full(model.n_rows, np.inf)
for i, s in enumerate(model.sense):
    if s == '=': lo[i] = hi[i] = model.b[i]
    elif s == '<=': hi[i] = model.b[i]
    elif s == '>=': lo[i] = model.b[i]
constraints = LinearConstraint(csr_matrix(A), lo, hi)
bounds = Bounds(model.lb, [1e9 if np.isinf(u) else u for u in model.ub])
integrality = np.array([1 if k == 'y' else 0 for k in model.var_kind])

t0 = time.time()
res = milp(model.c, constraints=constraints, integrality=integrality, bounds=bounds,
           options=dict(time_limit=120, mip_rel_gap=1e-6, disp=False))
sec = time.time() - t0
obj = float(res.fun)
bound = getattr(res, 'mip_dual_bound', None)
bound = float(bound) if bound is not None else obj
gap = max(0.0, (obj - bound) / abs(obj))
out = dict(instance='cp34_instance(12,B,2), M=35', objective=obj, bound=bound,
           solver_gap=gap, proven=bool(gap <= 1e-6), seconds=round(sec, 1))
print(json.dumps(out, indent=2))
json.dump(out, open('results/final_exact_m35.json', 'w'), indent=2)
