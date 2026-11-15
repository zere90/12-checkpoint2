"""Block 3: LP relaxation + our own rounding rule -> an integer start for each scaled size.

Rounding rule (ours, stated in the report): solve the LP relaxation of the reference B2 model
(src/model/build.py / src/lp/solve.py) on the scaled instance. For each VM k, look at its row of
fractional y[k,l] values over servers l allowed by its zone. Assign VM k to the allowed server
with the largest LP value y[k,l] (ties broken by lowest current projected load, to spread VMs
across equally-attractive servers instead of piling them all on one). This is a simple
"highest-fraction" rounding, the natural one for an assignment-type LP: it never needs the
zone/capacity structure beyond what the LP already encodes in y, and it always produces a
structurally zone-valid assignment (forbidden pairs have ub=0 in the LP, so their y is always 0
and is never picked over an allowed one, since every VM has at least one allowed server with
positive capacity in these instances).
"""
import json, sys, time
import numpy as np

sys.path.insert(0, '.')
import checkpoint4_generator_PUBLIC as gen
from src.model.build import build_b2_model
from src.lp.solve import solve_lp


def round_from_lp(d):
    model = build_b2_model(d)
    t0 = time.time()
    res = solve_lp(model)
    dt = time.time() - t0
    S, M = d['S'], d['M']
    y = res.x[: M * S].reshape(M, S)
    zone = np.asarray(d['zone'])
    req = np.asarray(d['vm_required_zone'])
    load = np.zeros(S)  # running projected CPU load share, for tie-breaking
    assign = np.empty(M, dtype=int)
    cpu_req = np.asarray(d['cpu_req'], dtype=float)
    cpu_cap = np.asarray(d['cpu_cap'], dtype=float)
    for k in range(M):
        allowed = np.where(zone == req[k])[0] if req[k] != -1 else np.arange(S)
        vals = y[k, allowed]
        best_val = vals.max()
        # tie-break: among (near-)ties on the LP value, pick the least-loaded allowed server
        tol = 1e-9
        tied = allowed[vals >= best_val - tol]
        chosen = tied[np.argmin(load[tied] / cpu_cap[tied])]
        assign[k] = chosen
        load[chosen] += cpu_req[k]
    return assign.tolist(), float(res.z), dt, res.n_fractional


if __name__ == '__main__':
    out = {}
    for size in (100, 200, 400):
        d = gen.scaled_instance('B', size, 2)
        assign, z_lp, dt, n_frac = round_from_lp(d)
        out[size] = dict(assign=assign, z_lp=z_lp, lp_seconds=round(dt, 2), n_fractional=n_frac)
        print(f'size {size}: z_LP={z_lp:.6f}  LP solve {dt:.2f}s  fractional y vars: {n_frac}')
        json.dump(dict(assign=assign), open(f'rounded_B_{size}.json', 'w'))
    json.dump(out, open('rounded_starts_summary.json', 'w'), indent=2)
