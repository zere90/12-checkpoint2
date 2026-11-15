"""Block 2, Figure F1: best-so-far cost vs evaluations, all four methods, seeds 0-9, mean with
min/max band. Reads the per-seed traces written by build_f1_cp4.py (f1_cp4_traces.json) and the
team's own instance (cp34_instance(12,'B',2)) exact/LP values for the reference lines.
Regenerates results/final_F1.png.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

Z_REF = 5.268322
Z_LP = 5.209695

traces = json.load(open('f1_cp4_traces.json'))

COLORS = {
    'SA, transfer+swap': 'tab:green',
    'SA-solo, transfer': 'tab:purple',
    'GA, penalty': 'tab:blue',
    'random search': 'tab:red',
}

fig, ax = plt.subplots(figsize=(7, 5))
# Common logging floor: SA logs from eval 1000, GA/random from eval ~1, with an `inf` placeholder
# until their first feasible solution. Start the shared grid at 1000 evals (every method has a
# real, finite best-so-far value by then) so the pre-first-improvement transient of one method
# doesn't dominate the y-scale of the whole figure.
GRID_START = 1000
for name, color in COLORS.items():
    runs = traces[name]  # list of 10 traces, each a list of (evals, best_so_far) pairs
    grid = np.linspace(GRID_START, 100000, 400)
    curves = []
    for tr in runs:
        pts = [(x, y) for x, y in tr if y != float('inf')]
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        curves.append(np.interp(grid, xs, ys, left=ys[0]))
    curves = np.array(curves)
    mean = curves.mean(axis=0)
    lo = curves.min(axis=0)
    hi = curves.max(axis=0)
    ax.plot(grid, mean, label=name, color=color)
    ax.fill_between(grid, lo, hi, color=color, alpha=0.15)

ax.axhline(Z_REF, color='black', linestyle='--', linewidth=1, label=r'$z_{\mathrm{ref}}$ = %.6f' % Z_REF)
ax.axhline(Z_LP, color='gray', linestyle=':', linewidth=1, label=r'$z_{\mathrm{LP}}$ = %.6f' % Z_LP)
ax.set_xlabel('evaluations')
ax.set_ylabel('best feasible cost so far')
ax.set_title('Team T12: best-so-far cost, all methods (seeds 0-9)')
ax.set_ylim(5.1, 6.1)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig('results/final_F1.png', dpi=150)
print('wrote results/final_F1.png')
