"""Block 3, Figure F2: exact-solver (HiGHS) wall time against scaled size M, with the 120s cap
marked and each bar labeled proven / not proven + solver gap. Reads results/final_exact_scaling.json.
Regenerates results/final_F2.png.
"""
import json
import matplotlib.pyplot as plt

data = json.load(open('results/final_exact_scaling.json'))
sizes = sorted(int(k) for k in data)
times = [data[str(s)]['seconds'] for s in sizes]
proven = [data[str(s)]['proven'] for s in sizes]
gaps = [data[str(s)]['solver_gap'] for s in sizes]

colors = ['tab:green' if p else 'tab:red' for p in proven]

fig, ax = plt.subplots(figsize=(6.5, 5))
bars = ax.bar([str(s) for s in sizes], times, color=colors)
ax.axhline(120, color='black', linestyle='--', linewidth=1, label='120 s limit')
for bar, p, g in zip(bars, proven, gaps):
    label = 'proven' if p else f'not proven\ngap {g*100:.2f}%'
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, label,
            ha='center', va='bottom', fontsize=8)
ax.set_xlabel('M (virtual machines)')
ax.set_ylabel('wall time, s')
ax.set_title('Exact solver (HiGHS) time against size')
ax.set_ylim(0, 145)
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig('results/final_F2.png', dpi=150)
print('wrote results/final_F2.png')
