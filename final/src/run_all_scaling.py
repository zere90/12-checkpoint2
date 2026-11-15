import csv, json, os, sys, time, importlib
from multiprocessing import Pool
sys.path.insert(0, '.')
import checkpoint4_generator_PUBLIC as gen
from src.meta.interface import check

TEAM = 'T12'
SIZES = (100, 200, 400)
SEEDS = list(range(10))
BUDGET = 100000
OUT = 'results/scaling_runs.csv'
COLS = ['team', 'configuration', 'story', 'variant', 'size', 'seed', 'budget', 'cost', 'evals', 'feasible', 'seconds', 'start', 'params']

os.makedirs('results', exist_ok=True)

rounded = {s: json.load(open(f'rounded_B_{s}.json'))['assign'] for s in SIZES}


def run_one(args):
    module_name, config_label, size, seed, start_label = args
    d = gen.scaled_instance('B', size, 2)
    start = rounded[size] if start_label == 'lp_rounded' else None
    solve = importlib.import_module(module_name).solve
    t0 = time.time()
    res = solve(d, 2, seed, BUDGET, params=None, start=start)
    sec = time.time() - t0
    cost, problems = check(d, 2, res['solution'])
    feasible = len(problems) == 0
    row = [TEAM, config_label, 'B', 2, size, seed, BUDGET, f'{cost:.9f}', res.get('evals'), feasible, f'{sec:.2f}', start_label, '']
    return row


def write_rows(rows):
    new = not os.path.exists(OUT)
    with open(OUT, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLS)
        for r in rows:
            w.writerow(r)
            fh.flush()


FAST_JOBS = []   # SA, SA-solo: run inline, both starts
for size in SIZES:
    for seed in SEEDS:
        FAST_JOBS.append(('src.meta.sa', 'SA, own start', size, seed, 'own'))
        FAST_JOBS.append(('src.meta.sa', 'SA, LP-rounded start', size, seed, 'lp_rounded'))
        FAST_JOBS.append(('src.meta.sa_solo', 'SA-solo, own start', size, seed, 'own'))
        FAST_JOBS.append(('src.meta.sa_solo', 'SA-solo, LP-rounded start', size, seed, 'lp_rounded'))

SLOW_JOBS = []   # GA, random search: start has no effect by design -- run once, label 'own' (= 'lp_rounded', identical)
for size in SIZES:
    for seed in SEEDS:
        SLOW_JOBS.append(('src.meta.ga', 'GA, penalty', size, seed, 'own'))
        SLOW_JOBS.append(('src.meta.random_search', 'random search', size, seed, 'own'))

if __name__ == '__main__':
    print(f'{len(FAST_JOBS)} fast jobs (SA / SA-solo), {len(SLOW_JOBS)} slow jobs (GA / random)', flush=True)
    t0 = time.time()
    rows = [run_one(j) for j in FAST_JOBS]
    write_rows(rows)
    print(f'fast jobs done in {time.time()-t0:.1f}s', flush=True)

    t0 = time.time()
    with Pool(2) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, SLOW_JOBS)):
            write_rows([row])
            print(f'[{i+1}/{len(SLOW_JOBS)}] {row[1]} size={row[4]} seed={row[5]} cost={row[7]} feasible={row[9]} {row[10]}s  (elapsed {time.time()-t0:.0f}s)', flush=True)
    print(f'ALL DONE in {time.time()-t0:.1f}s', flush=True)
