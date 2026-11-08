"""Manual tuning grid of the SA-solo role: 2 parameters, 3 values each, seeds 100-104 only.

    python experiments/tune_sa_solo.py

T0 and Tend are the rule-based values (random walk of 3000 proposals, measured once with seed 100)
times 1/3, 1 or 3. The cell with the lowest mean cost is kept; on a tie the rule-based cell wins.
Writes results/noken_tuning.csv, results/noken_tuning_summary.csv and results/noken_params.json.
The reference values are not used here.
"""
import csv
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import checkpoint4_generator_PUBLIC as gen
from src.meta.interface import check
from src.meta.sa_solo import rule_temperatures, solve

SEEDS = list(range(100, 105))
BUDGET = 100000
FACTORS = (1.0, 1.0 / 3.0, 3.0)      # 1.0 first: the rule-based cell wins a tie
COLS = ['team', 'role', 'configuration', 'seed', 'budget', 'cost', 'evals', 'feasible', 'seconds', 'params']


def main():
    d = gen.cp34_instance(12, 'B', 2)
    rule = rule_temperatures(d, 2, 100)
    print('rule-based values', rule)
    results = os.path.join(ROOT, 'results')
    summary, best = [], None
    with open(os.path.join(results, 'noken_tuning.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for ft0 in FACTORS:
            for ftend in FACTORS:
                params = {'T0': rule['T0'] * ft0, 'Tend': rule['Tend'] * ftend}
                costs, feasible = [], 0
                for seed in SEEDS:
                    t0 = time.time()
                    res = solve(d, 2, seed, BUDGET, params=params)
                    sec = time.time() - t0
                    cost, problems = check(d, 2, res['solution'])
                    ok = not problems
                    feasible += ok
                    costs.append(cost)
                    w.writerow(['T12', 'SA-solo', 'SA, transfer', seed, BUDGET, f'{cost:.9f}', res['evals'], ok, f'{sec:.2f}', json.dumps(params)])
                row = {'T0_factor': round(ft0, 4), 'Tend_factor': round(ftend, 4), 'T0': params['T0'], 'Tend': params['Tend'],
                       'mean_cost': statistics.mean(costs), 'best': min(costs), 'worst': max(costs),
                       'sd': statistics.stdev(costs), 'feasible': feasible}
                summary.append(row)
                print(row['T0_factor'], row['Tend_factor'], round(row['mean_cost'], 6))
                if feasible == len(SEEDS) and (best is None or row['mean_cost'] < best['mean_cost']):
                    best = row
    with open(os.path.join(results, 'noken_tuning_summary.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(results, 'noken_params.json'), 'w') as fh:
        json.dump({'rule': rule, 'selected': best, 'params': {'T0': best['T0'], 'Tend': best['Tend']}}, fh, indent=2)
    print('selected', best['T0'], best['Tend'], best['mean_cost'])


if __name__ == '__main__':
    main()
