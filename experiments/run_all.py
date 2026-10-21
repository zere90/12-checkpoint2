"""experiments/run_all.py -- M5 (Zere, project lead): ties every role's module together and
regenerates every results/*.csv for Checkpoint 3's Tables 1-3 and the active-rows breakdown.

Run from the repo root: python3 -m experiments.run_all
"""
import csv
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from checkpoint3_generator_PUBLIC import cp34_instance
from model.build import build_b2_model
from lp.solve import solve_lp
from kkt.checker import check_model
from sens.perturb import rhs_sensitivity
from bound import gap  # noqa: F401 (imported for parity with the role; not used standalone here)

SEED = 12
RESULTS_DIR = os.path.join(_ROOT, "results")


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    d = cp34_instance(SEED, 'B', 2)
    model = build_b2_model(d)

    # --- checks.csv: instance check values vs. the team's task file ---
    checks = [
        ("S (servers)", d['S'], 10),
        ("M (VMs)", d['M'], 35),
        ("sum cpu_cap", int(d['cpu_cap'].sum()), 454),
        ("sum mem_cap", int(d['mem_cap'].sum()), 1680),
        ("sum cpu_req", int(d['cpu_req'].sum()), 172),
        ("sum mem_req", int(d['mem_req'].sum()), 615),
        ("forbidden pairs (fixed to 0)", model.meta['n_forbidden'], 60),
        ("variables (Task 1)", model.n_vars, 360),
        ("constraint rows (Task 1)", model.n_rows, 95),
    ]
    _write_csv(
        os.path.join(RESULTS_DIR, "checks.csv"),
        ["quantity", "computed", "task_file", "match"],
        [(q, c, t, c == t) for q, c, t in checks],
    )
    print("checks.csv:")
    for q, c, t in checks:
        print(f"  {q}: computed={c} task_file={t} match={c == t}")

    # --- solve the LP relaxation ---
    r = solve_lp(model)
    print(f"\nz_LP = {r.z:.6f}  (solver={r.solver}, time={r.time_sec:.6f}s)")
    print(f"fractional y: {r.n_fractional}")

    _write_csv(
        os.path.join(RESULTS_DIR, "table1.csv"),
        ["quantity", "value"],
        [
            ("n_variables", model.n_vars),
            ("n_constraint_rows", model.n_rows),
            ("n_forbidden_fixed", model.meta['n_forbidden']),
            ("z_LP", f"{r.z:.6f}"),
            ("solver", r.solver),
            ("time_sec", f"{r.time_sec:.6f}"),
            ("n_fractional_y", r.n_fractional),
        ],
    )

    frac_rows = []
    for label, val in r.fractional_vars:
        k_str, l_str = label[2:-1].split(',')
        frac_rows.append((label, f"{val:.6f}", k_str, l_str))
    _write_csv(
        os.path.join(RESULTS_DIR, "table1_fractional.csv"),
        ["variable", "value", "vm_k", "server_l"],
        frac_rows,
    )

    # --- Table 2: KKT conditions on our own solution ---
    rep = check_model(model, r)
    print("\nKKT conditions:")
    print(f"  (a) primal feasibility: {rep.primal_feasibility:.3e}  satisfied={rep.primal_feasibility <= rep.tol}")
    print(f"  (b) dual feasibility:   {rep.dual_feasibility:.3e}  satisfied={rep.dual_feasibility <= rep.tol}")
    print(f"  (c) compl. slackness:   {rep.complementary_slackness:.3e}  satisfied={rep.complementary_slackness <= rep.tol}")
    print(f"  (d) reduced costs:      {rep.reduced_costs:.3e}  satisfied={rep.reduced_costs <= rep.tol}")
    print(f"  (e) strong duality:     {rep.duality_gap:.3e}  satisfied={abs(rep.duality_gap) <= rep.tol}")
    print(f"  ALL SATISFIED: {rep.all_satisfied}")

    _write_csv(
        os.path.join(RESULTS_DIR, "table2.csv"),
        ["condition", "value", "satisfied (<=1e-6)"],
        [
            ("(a) primal feasibility, max violation", f"{rep.primal_feasibility:.3e}", rep.primal_feasibility <= rep.tol),
            ("(b) dual feasibility (signs), max violation", f"{rep.dual_feasibility:.3e}", rep.dual_feasibility <= rep.tol),
            ("(c) complementary slackness, max |pi*slack|", f"{rep.complementary_slackness:.3e}", rep.complementary_slackness <= rep.tol),
            ("(d) reduced costs, max violation", f"{rep.reduced_costs:.3e}", rep.reduced_costs <= rep.tol),
            ("(e) strong duality, |gap|", f"{rep.duality_gap:.3e}", abs(rep.duality_gap) <= rep.tol),
            ("ALL FIVE satisfied", "", rep.all_satisfied),
        ],
    )

    # --- Table 3: sensitivity of the designated constraint, (B1) for VM k*=13 ---
    k_star = 13
    designated_row = None
    for i, (fam, lab) in enumerate(zip(model.row_family, model.row_label)):
        if fam == 'B1' and f"VM {k_star}" in lab:
            designated_row = i
            break
    assert designated_row is not None, "designated row (B1, VM 13) not found"

    s = rhs_sensitivity(model, designated_row, delta=0.01)
    print(f"\nSensitivity of row {designated_row} ({model.row_label[designated_row]}):")
    print(f"  left_slope={s.left_slope:.6f}  pi={s.pi:.6f}  right_slope={s.right_slope:.6f}")
    print(f"  interval_ok={s.interval_ok}  degenerate={s.degenerate}")

    _write_csv(
        os.path.join(RESULTS_DIR, "table3.csv"),
        ["quantity", "value"],
        [
            ("row", model.row_label[designated_row]),
            ("b0", s.b0),
            (f"z(b-{s.delta})", f"{s.z_minus:.6f}"),
            ("z(b)", f"{s.z0:.6f}"),
            (f"z(b+{s.delta})", f"{s.z_plus:.6f}"),
            ("left_slope", f"{s.left_slope:.6f}"),
            ("pi (solver dual price)", f"{s.pi:.6f}"),
            ("right_slope", f"{s.right_slope:.6f}"),
            ("interval_ok (left<=pi<=right)", s.interval_ok),
            ("degenerate", s.degenerate),
        ],
    )

    # --- Active rows per constraint family (Task 5 / S5) ---
    row_val = model.A @ r.x
    active = {}
    total = {}
    TOL = 1e-6
    for i, fam in enumerate(model.row_family):
        total[fam] = total.get(fam, 0) + 1
        sense = model.sense[i]
        b_i = model.b[i]
        is_tight = abs(row_val[i] - b_i) <= TOL
        if is_tight:
            active[fam] = active.get(fam, 0) + 1
        else:
            active.setdefault(fam, active.get(fam, 0))

    families = sorted(total.keys())
    print("\nActive rows per family:")
    for fam in families:
        a = active.get(fam, 0)
        t = total[fam]
        print(f"  {fam}: {a}/{t}")

    _write_csv(
        os.path.join(RESULTS_DIR, "active_rows.csv"),
        ["family", "active", "total"],
        [(fam, active.get(fam, 0), total[fam]) for fam in families],
    )

    print("\nAll results/*.csv regenerated.")


if __name__ == "__main__":
    main()
