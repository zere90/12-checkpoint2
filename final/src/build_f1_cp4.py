import sys, json, time, os
sys.path.insert(0, '.')
import checkpoint4_generator_PUBLIC as gen
from src.meta.ga import solve as ga_solve
from src.meta.sa import solve as sa_solve
from src.meta.sa_solo import solve as solo_solve
from src.meta.random_search import solve as rand_solve

d = gen.cp34_instance(12, 'B', 2)
BUDGET = 100000
OUT = 'f1_cp4_traces.json'
out = json.load(open(OUT)) if os.path.exists(OUT) else {}

for name, fn, params in [
    ('SA, transfer+swap', sa_solve, {"T0": 0.820689, "Tend": 0.000795}),
    ('SA-solo, transfer', solo_solve, {"T0": 0.29006072102002334, "Tend": 0.0010197932600936345}),
    ('GA, penalty', ga_solve, None),
    ('random search', rand_solve, None),
]:
    if name in out:
        print(name, 'already done, skipping', flush=True)
        continue
    traces = []
    t0 = time.time()
    for seed in range(10):
        res = fn(d, 2, seed, BUDGET, params=params, start=None)
        traces.append(res['trace'])
    print(name, 'done in', round(time.time() - t0, 1), 's', flush=True)
    out[name] = traces
    json.dump(out, open(OUT, 'w'))

print('ALL DONE')
