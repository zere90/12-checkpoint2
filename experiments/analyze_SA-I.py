"""Analysis of the final SA runs (role SA-I): Table 1, Table 2, Figure F1, acceptance-vs-temperature curve.

    python experiments/analyze_SA-I.py --runs results/tilesbay_runs.csv [--baseline results/<file>.csv] [--params '{"T0":..,"Tend":..}']

The runs are reproduced with solve (deterministic given the seed) to get the traces and the acceptance log;
the costs are compared with the results file. z_ref and z_LP are used ONLY here, for the gaps (never in solve).
"""
import argparse, csv, json, os, statistics, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import checkpoint4_generator_PUBLIC as gen
from src.meta import sa

Z_REF, Z_LP = 5.268322, 5.209695          # task file of team T12 (6 decimals)
INSTANCE_SEED, VARIANT = 12, 2


def gaps(z):
    return 100 * (z - Z_REF) / Z_REF, 100 * (z - Z_LP) / z


def improving_moves(d, assign):
    """Counts of strictly improving feasible transfers and swaps from `assign` (checker-based, all moves)."""
    base, prob = gen.evaluate_B(d, assign)
    assert prob == []
    M, S = d['M'], d['S']
    nt = ns = 0
    for k in range(M):
        for t in range(S):
            if t == assign[k]:
                continue
            a = list(assign); a[k] = t
            c, p = gen.evaluate_B(d, a)
            if not p and c < base - 1e-12:
                nt += 1
    for i in range(M):
        for j in range(i + 1, M):
            if assign[i] == assign[j]:
                continue
            a = list(assign); a[i], a[j] = a[j], a[i]
            c, p = gen.evaluate_B(d, a)
            if not p and c < base - 1e-12:
                ns += 1
    return nt, ns


def share(info, lo, hi):
    """Share of accepted proposals among the worsening ones with proposal index in [lo, hi)."""
    idx = np.array(info['worse_idx']); acc = np.array(info['worse_acc'])
    m = (idx >= lo) & (idx < hi)
    return (float(acc[m].sum()) / int(m.sum()) if m.sum() else float('nan')), int(m.sum()), int(acc[m].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='results/tilesbay_runs.csv')
    ap.add_argument('--params', default=None)
    ap.add_argument('--baseline', default=None, help='results file of the random-search baseline (owner: SA-T)')
    ap.add_argument('--outdir', default='results')
    a = ap.parse_args()
    params = json.loads(a.params) if a.params else None
    d = gen.cp34_instance(INSTANCE_SEED, 'B', VARIANT)
    rows = list(csv.DictReader(open(a.runs)))
    seeds = [int(r['seed']) for r in rows]
    budget = int(rows[0]['budget'])
    out = {}

    # ---- rerun, compare with the file
    res = {}
    for r in rows:
        s = int(r['seed'])
        res[s] = sa.solve(d, VARIANT, s, budget, params=params)
        assert abs(res[s]['cost'] - float(r['cost'])) < 1e-6, f'seed {s}: rerun differs from the file'
    costs = [float(r['cost']) for r in rows]
    feas = [r['feasible'] == 'True' for r in rows]
    fc = [c for c, f in zip(costs, feas) if f]
    secs = [float(r['seconds']) for r in rows]

    # ---- Table 1
    stat = {'best': min(fc), 'median': statistics.median(fc), 'mean': statistics.fmean(fc), 'worst': max(fc)}
    stat['std'] = statistics.stdev(fc) if len(fc) > 1 else 0.0
    out['table1'] = {'cost': stat, 'feasible': f'{sum(feas)}/{len(feas)}', 'mean_seconds': statistics.fmean(secs),
                     'gap_ref': {k: gaps(stat[k])[0] for k in ('best', 'median', 'mean', 'worst')},
                     'gap_LP': {k: gaps(stat[k])[1] for k in ('best', 'median', 'mean', 'worst')},
                     'gap_LP_of_z_ref (gap of the relaxation itself)': 100 * (Z_REF - Z_LP) / Z_REF}

    # ---- diversity of final solutions, and where each curve becomes flat
    finals = {s: tuple(res[s]['solution']) for s in seeds}
    out['distinct_final_assignments'] = len(set(finals.values()))
    flat = {}
    for s in seeds:
        tr = res[s]['trace']
        final = tr[-1][1]
        first = next(e for e, c in tr if c <= final + 1e-12)
        flat[s] = first
    out['evals_until_final_value_first_reached'] = flat
    out['share_of_budget_flat_from (median/max)'] = [statistics.median(flat.values()) / budget, max(flat.values()) / budget]

    # ---- Table 2: worst seed = feasible run with the largest cost (ties: smallest seed)
    worst = max((s for s, f in zip(seeds, feas) if f), key=lambda s: (round(costs[seeds.index(s)], 9), -s))
    info = res[worst]['info']
    nt, ns = improving_moves(d, res[worst]['solution'])
    K = info['n_proposals']
    cut = int(0.05 * K)
    f5, n5, a5 = share(info, 0, cut)
    l5, nl5, al5 = share(info, K - cut, K)
    out['table2'] = {'worst_seed': worst, 'final_cost': costs[seeds.index(worst)], 'improving_transfers': nt, 'improving_swaps': ns,
                     'accept_share_worsening_first5pct': f5, 'worsening_first5pct (n, accepted)': [n5, a5],
                     'accept_share_worsening_last5pct': l5, 'worsening_last5pct (n, accepted)': [nl5, al5],
                     'annealing_proposals K': K, 'T0': info['T0'], 'Tend': info['Tend'], 'walk_evals': info['walk'],
                     'infeasible_among_worsening (worst seed)': int(len(info['worse_feas']) - sum(info['worse_feas']))}
    # same numbers for all seeds (to see whether the choice of the tied worst seed matters)
    allshare = {}
    for s in seeds:
        i = res[s]['info']; k = i['n_proposals']; c = int(0.05 * k)
        allshare[s] = [share(i, 0, c)[0], share(i, k - c, k)[0]]
    out['accept_share_first_last_5pct_all_seeds'] = allshare
    out['total_accept_rate_all_proposals'] = {s: res[s]['info']['n_accepted'] / res[s]['info']['n_proposals'] for s in seeds}

    # ---- optional baseline
    if a.baseline and os.path.exists(a.baseline):
        br = [float(r['cost']) for r in csv.DictReader(open(a.baseline)) if r['feasible'] == 'True']
        out['baseline_random'] = {'n_feasible': len(br), 'best': min(br), 'median': statistics.median(br), 'mean': statistics.fmean(br),
                                  'worst': max(br), 'gap_ref_mean': gaps(statistics.fmean(br))[0], 'gap_LP_mean': gaps(statistics.fmean(br))[1]}

    # ---- F1: mean best-so-far with min-max band
    ev = np.array([e for e, _ in res[seeds[0]]['trace']])
    T = np.array([[c for _, c in res[s]['trace']] for s in seeds])
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.fill_between(ev, T.min(0), T.max(0), alpha=0.25, label='min-max (10 runs)')
    ax.plot(ev, T.mean(0), lw=1.6, label='mean best-so-far')
    ax.axhline(Z_REF, color='k', ls='--', lw=1, label=f'z_ref = {Z_REF}')
    ax.axhline(Z_LP, color='r', ls=':', lw=1.2, label=f'z_LP = {Z_LP}')
    ax.set_xlabel('evaluations'); ax.set_ylabel('best feasible cost so far'); ax.set_ylim(Z_LP - 0.03, 6.3)
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, 'F1_SA-I.png'), dpi=200); plt.close(fig)
    # zoomed version (last part) for the text
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    m = ev <= 0.4 * budget
    ax.fill_between(ev[m], T.min(0)[m], T.max(0)[m], alpha=0.25); ax.plot(ev[m], T.mean(0)[m])
    ax.axhline(Z_REF, color='k', ls='--', lw=1); ax.axhline(Z_LP, color='r', ls=':', lw=1.2)
    ax.set_xlabel('evaluations'); ax.set_ylabel('best feasible cost'); ax.set_ylim(Z_LP - 0.02, Z_REF + 0.5)
    fig.tight_layout(); fig.savefig(os.path.join(a.outdir, 'F1_SA-I_zoom.png'), dpi=200); plt.close(fig)

    # ---- acceptance share of worsening moves vs temperature (all 10 seeds pooled, log-spaced bins)
    Ts = np.concatenate([res[s]['info']['worse_T'] for s in seeds]); Ac = np.concatenate([res[s]['info']['worse_acc'] for s in seeds]).astype(float)
    edges = np.logspace(np.log10(Ts.min()), np.log10(Ts.max()), 25)
    mid, shr, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (Ts >= lo) & (Ts <= hi)
        if mk.sum() >= 20:
            mid.append(np.sqrt(lo * hi)); shr.append(Ac[mk].mean()); cnt.append(int(mk.sum()))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.semilogx(mid, shr, 'o-', ms=3); ax.invert_xaxis()
    ax.set_xlabel('temperature T (decreasing to the right)'); ax.set_ylabel('accepted / worsening proposals'); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout(); fig.savefig(os.path.join(a.outdir, 'accept_vs_T_SA-I.png'), dpi=200); plt.close(fig)
    out['accept_vs_T'] = {'T': [float(x) for x in mid], 'share': [float(x) for x in shr], 'n': cnt}

    json.dump(out, open(os.path.join(a.outdir, 'tilesbay_analysis.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == '__main__':
    main()
