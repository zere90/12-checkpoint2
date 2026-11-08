"""Analysis of the SA-solo final runs: Table 1, Table 2, Figure F1, the acceptance curve, the worst-seed checks.

    python experiments/analyze_sa_solo.py

Reads results/noken_runs.csv, reruns the ten seeds (solve is deterministic), stops if a rerun disagrees with the
file, and writes results/noken_analysis.json, results/F1_noken.png and results/accept_vs_T_noken.png.
The reference values are used here only, for reporting.
"""
import csv
import itertools
import json
import os
import statistics as st
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import checkpoint4_generator_PUBLIC as gen
from src.meta.interface import check
from src.meta.sa_solo import solve

Z_LP, Z_REF = 5.209695, 5.268322
RESULTS = os.path.join(ROOT, 'results')
TOL = 1e-9
INK, MUTED, GRID, BLUE = '#1f2328', '#5f6670', '#e3e6ea', '#2a6fdb'


def gap_ref(z):
    return 100.0 * (z - Z_REF) / Z_REF


def gap_lp(z):
    return 100.0 * (z - Z_LP) / z


def summary(rows):
    feas = [float(r['cost']) for r in rows if r['feasible'] == 'True']
    stats = {'best': min(feas), 'median': st.median(feas), 'mean': st.mean(feas), 'worst': max(feas)}
    return {'cost': stats, 'std': st.stdev(feas) if len(feas) > 1 else 0.0, 'feasible': len(feas), 'runs': len(rows),
            'mean_seconds': st.mean(float(r['seconds']) for r in rows),
            'gap_ref': {k: gap_ref(v) for k, v in stats.items()}, 'gap_LP': {k: gap_lp(v) for k, v in stats.items()}}


def moves_from(d, assign):
    """Every transfer and every swap from `assign`, judged by the checker. Improving = feasible and cheaper by more than TOL."""
    base = check(d, 2, assign)[0]
    S, M = d['S'], d['M']
    transfers, swaps = [], []
    n_transfer = n_swap = 0
    equal_transfer = equal_swap = 0
    for k, l in itertools.product(range(M), range(S)):
        if l == assign[k]:
            continue
        n_transfer += 1
        new = list(assign)
        new[k] = l
        cost, problems = check(d, 2, new)
        if not problems:
            if cost < base - TOL:
                transfers.append((cost - base, k, l))
            elif abs(cost - base) <= TOL:
                equal_transfer += 1
    for i, j in itertools.combinations(range(M), 2):
        if assign[i] == assign[j]:
            continue
        n_swap += 1
        new = list(assign)
        new[i], new[j] = new[j], new[i]
        cost, problems = check(d, 2, new)
        if not problems:
            if cost < base - TOL:
                swaps.append((cost - base, i, j))
            elif abs(cost - base) <= TOL:
                equal_swap += 1
    return {'tried_transfers': n_transfer, 'tried_swaps': n_swap, 'improving_transfers': len(transfers), 'improving_swaps': len(swaps),
            'equal_cost_transfers': equal_transfer, 'equal_cost_swaps': equal_swap,
            'best_transfer': min(transfers) if transfers else None, 'best_swap': min(swaps) if swaps else None}


def swap_barrier(d, assign, i, j):
    """A swap done by two transfers: the cost change of the first transfer, for both orders."""
    base = check(d, 2, assign)[0]
    out = []
    for a, b in ((i, j), (j, i)):
        new = list(assign)
        new[a] = assign[b]
        cost, problems = check(d, 2, new)
        out.append({'first_vm': a, 'to_server': int(assign[b]), 'delta': cost - base, 'feasible': not problems})
    return out


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    with open(os.path.join(RESULTS, 'noken_runs.csv'), newline='') as fh:
        rows = list(csv.DictReader(fh))
    sa_rows = [r for r in rows if r['configuration'] == 'SA, transfer']
    random_rows = [r for r in rows if r['configuration'] == 'random']
    assert [int(r['seed']) for r in sa_rows] == list(range(10)) and [int(r['seed']) for r in random_rows] == list(range(10))
    d = gen.cp34_instance(12, 'B', 2)
    params = json.loads(sa_rows[0]['params'])

    runs = {}
    for r in sa_rows:
        seed = int(r['seed'])
        res = solve(d, 2, seed, int(r['budget']), params=params)
        assert abs(res['cost'] - float(r['cost'])) < 1e-6, f'seed {seed}: rerun differs from the file'
        runs[seed] = res

    out = {'params': params, 'table1': summary(sa_rows), 'random': summary(random_rows)}

    evals = np.array([e for e, _ in runs[0]['trace']])
    curves = np.array([[b for _, b in runs[s]['trace']] for s in range(10)])
    final_at = {}
    for s in range(10):
        c = curves[s]
        final_at[s] = int(evals[int(np.argmax(c <= c[-1] + TOL))])
    out['evals_until_final_value'] = final_at
    out['median_evals_until_final_value'] = float(np.median(list(final_at.values())))
    out['costs_by_seed'] = {s: runs[s]['cost'] for s in range(10)}
    out['reached_zref'] = int(sum(runs[s]['cost'] <= Z_REF + 1e-5 for s in range(10)))
    out['distinct_final_costs'] = len({round(runs[s]['cost'], 6) for s in range(10)})

    # worst seed: the largest cost, the smallest seed on a tie
    worst = max(range(10), key=lambda s: (round(runs[s]['cost'], 9), -s))
    res = runs[worst]
    diag = res['diag']
    moves = moves_from(d, res['solution'])
    out['table2'] = {'seed': worst, 'final_cost': res['cost'], 'improving_transfers': moves['improving_transfers'], 'improving_swaps': moves['improving_swaps'],
                     'worse_first': diag['worse_first'], 'accepted_first': diag['accepted_first'],
                     'share_first': diag['accepted_first'] / diag['worse_first'],
                     'worse_last': diag['worse_last'], 'accepted_last': diag['accepted_last'],
                     'share_last': diag['accepted_last'] / diag['worse_last']}
    out['worst_seed'] = {'moves': moves, 'final_value_at': final_at[worst], 'proposals': diag['proposals'], 'Tend': diag['Tend'],
                         'gap_ref': gap_ref(res['cost']), 'gap_LP': gap_lp(res['cost']),
                         'rejected_capacity_share': diag['rejected_capacity'] / diag['proposals']}
    if moves['best_swap']:
        delta, i, j = moves['best_swap']
        out['worst_seed']['best_swap_detail'] = {'delta': delta, 'vm_i': i, 'vm_j': j, 'server_i': int(res['solution'][i]), 'server_j': int(res['solution'][j]),
                                                 'cpu_i': int(d['cpu_req'][i]), 'cpu_j': int(d['cpu_req'][j]),
                                                 'via_transfers': swap_barrier(d, res['solution'], i, j)}
    all_moves = {s: moves_from(d, runs[s]['solution']) for s in range(10)}
    out['moves_by_seed'] = {s: {k: all_moves[s][k] for k in ('improving_transfers', 'improving_swaps')} for s in range(10)}
    out['rejected_capacity_share_mean'] = float(np.mean([runs[s]['diag']['rejected_capacity'] / runs[s]['diag']['proposals'] for s in range(10)]))
    out['accept_first_last_by_seed'] = {s: [runs[s]['diag']['accepted_first'] / runs[s]['diag']['worse_first'],
                                            runs[s]['diag']['accepted_last'] / runs[s]['diag']['worse_last']] for s in range(10)}

    # acceptance against temperature: all ten seeds pooled, the 100 bins merged into 20
    worse = np.sum([runs[s]['diag']['bin_worse'] for s in range(10)], axis=0)
    acc = np.sum([runs[s]['diag']['bin_accepted'] for s in range(10)], axis=0)
    T_bin = np.array(runs[0]['diag']['bin_T'])
    groups = [list(range(i, i + 5)) for i in range(0, 100, 5)]
    T_g = [float(np.mean(T_bin[g])) for g in groups]
    share_g = [float(acc[g].sum() / worse[g].sum()) for g in groups]
    out['accept_vs_T'] = {'T': T_g, 'share': share_g}

    fig, ax = plt.subplots(figsize=(6.4, 2.7), dpi=200)
    ax.fill_between(evals, curves.min(axis=0), curves.max(axis=0), color=BLUE, alpha=0.18, linewidth=0, label='min-max of 10 runs')
    ax.plot(evals, curves.mean(axis=0), color=BLUE, linewidth=2, label='mean best-so-far')
    ax.axhline(Z_REF, color=INK, linestyle='--', linewidth=1, label=f'$z_{{ref}}$ = {Z_REF:.6f}')
    ax.axhline(Z_LP, color=MUTED, linestyle=':', linewidth=1.2, label=f'$z_{{LP}}$ = {Z_LP:.6f}')
    ax.set_xlim(0, evals[-1])
    ax.set_ylim(Z_LP - 0.03, 5.70)
    ax.set_xlabel('evaluations', fontsize=9, color=INK)
    ax.set_ylabel('best feasible cost', fontsize=9, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'F1_noken.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 2.2), dpi=200)
    ax.plot(T_g, share_g, color=BLUE, linewidth=2, marker='o', markersize=4)
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_ylim(0, 1)
    ax.set_xlabel('temperature T (log scale; the run moves from left to right)', fontsize=9, color=INK)
    ax.set_ylabel('accepted among\nworsening moves', fontsize=9, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'accept_vs_T_noken.png'))
    plt.close(fig)

    with open(os.path.join(RESULTS, 'noken_analysis.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ('table1', 'random', 'table2', 'worst_seed', 'moves_by_seed', 'evals_until_final_value', 'median_evals_until_final_value',
                                          'costs_by_seed', 'reached_zref', 'distinct_final_costs', 'rejected_capacity_share_mean', 'accept_first_last_by_seed')}))
    print('accept', [round(x, 3) for x in share_g], [round(x, 4) for x in T_g])


if __name__ == '__main__':
    main()
