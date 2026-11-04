# T12 — Checkpoint 2 (Unconstrained Component)

Team T12 · Storyline B (cloud resource allocation) · variant 2 · seed 12

## Roles

Chosen for a team of five, per Section 5 of the instructions (role table).
**Fill in real names before the first commit** — the placeholders below
must become the actual GitHub-linked names before anything is pushed.

| Role | Member | GitHub | Code authored (`src/`) | Analysis section | Hand trace |
|------|--------|--------|-------------------------|-------------------|------------|
| M1 — GD | Tilesbay | [@etilesbay-stack](https://github.com/etilesbay-stack) | `optim/gd.py`, `problems/rosenbrock.py`, `check_grad` | S1 — Step size and stability | H1 |
| M2 — Newton | Rakhima | [@queenbikeshhh](https://github.com/queenbikeshhh) | `optim/newton.py`, `problems/quadratic.py` | S2 — Conditioning, cost and safeguards | H2 |
| M3 — Momentum | Tamerlan | [@Tabix01](https://github.com/Tabix01) | `optim/momentum.py`, `experiments/plots.py` | S3 — Momentum in the valley | H3 |
| M4 — Adam | Danial | [@Haigatu](https://github.com/Haigatu) | `optim/adam.py` | S4 — Adam and the rotated axes | H4 |
| M5 — Project lead | Zere | [@zere90](https://github.com/zere90) | `problems/project.py`, `experiments/run_all.py` (tables), assembling the PDF | S5 — Project block | H5 |

This is our own proposed split; the instructions say "choose the roles
yourselves... write the choice into README.md in your first commit, and
keep it." Change the table above if your team wants a different split —
just do it *before* the first commit, not after.

## How to run

```
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib        # matplotlib only needed for plots.py
cd src
python -m experiments.run_all        # regenerates every results/*.csv
python -m experiments.plots          # regenerates results/figures/F1-F3
pytest ../tests -q                   # or: for f in ../tests/test_*.py; do python3 "$f"; done
```

`run_all.py` is deterministic (fixed starts, no randomness) and reproduces
every number behind Table 1 and Table 2, including the hyperparameter
grid search of Section 3. A full run takes a few minutes, mostly spent on
the grid searches for GD/momentum/Adam on R1 and R2 (tens of thousands of
GD iterations per grid cell near convergence).

## Environment

- Python 3.11
- `numpy` (solvers, problems) — the only dependency inside `src/optim/`
  and `src/problems/`, as required by Section 2 ("from scratch").
- `matplotlib` — plotting only (`experiments/plots.py`), as explicitly
  allowed by Section 2.
- `pytest` optional — the test files also run standalone with
  `python3 tests/test_X.py` (see each file's `if __name__ == "__main__"`
  block), no test framework required.
- No `scipy.optimize`, `sklearn`, autograd/JAX/PyTorch anywhere in
  `src/`.

## Repository layout

```
src/optim/        gd.py newton.py momentum.py adam.py _common.py
src/problems/      rosenbrock.py quadratic.py project.py
src/experiments/   run_all.py plots.py
tests/              test_gd.py test_newton.py test_momentum.py test_adam.py test_project.py
results/            *.csv (every number behind Table 1 and Table 2) + figures/F1-F3.png
hand/               reference derivations for H1-H5 (see hand/README.md)
T12_Checkpoint2.tex / .pdf   the submitted report
```

---

# Checkpoint 3 (Constrained Component)

Same repo, new checkpoint. Storyline B, **variant 2 (zones)**, seed 12. Tag for this checkpoint
will be `checkpoint3`.

## Roles

Same five members, new code per role. **This is our own proposed split** — change it before the
first Checkpoint-3 commit if the team wants something different.

| Role | Member | Code authored (`src/`, new for CP3) | Task | Signed section | Hand task |
|------|--------|--------------------------------------|------|-----------------|-----------|
| M1 | Tilesbay | `model/build.py` | Task 1 | S1 — The model and what it ignores | H1 |
| M2 | Rakhima | `lp/solve.py` | Task 2, Table 1 | S2 — Value and fractionality of the LP | H2 |
| M3 | Tamerlan | `kkt/checker.py` | Task 3, Table 2 | S3 — Is the solution optimal, and how do we know? | H3 |
| M4 | Danial | `sens/perturb.py` | Task 4, Table 3 | S4 — What the dual price says about the storyline | H4 |
| M5 (project lead) | Zere | `bound.py`, `experiments/run_all.py`, tables, assembling the PDF | Task 5 | S5 — The bound and the bridge to Checkpoint 4 | H5 |

## How to run (once everyone has pushed their module)

```
pip install numpy scipy --break-system-packages    # or inside your venv, no flag needed there
python3 -m experiments.run_all                       # regenerates results/*.csv for CP3
for f in tests/test_model.py tests/test_lp.py tests/test_kkt.py tests/test_sens.py tests/test_bound.py; do python3 "$f"; done
```

Each `tests/test_<role>.py` also runs completely standalone. Note: Checkpoint 3's
`experiments/run_all.py` lives at the **repo root** (not under `src/`, unlike Checkpoint 2's), and
it will overwrite `results/table1.csv`, `results/table2.csv` and `results/checks.csv` with CP3
numbers — the CP2 values stay permanently recoverable via `git show checkpoint2:results/table1.csv`
since the `checkpoint2` tag is immutable. This is expected, not a bug.

## Repository layout (new for Checkpoint 3)

```
checkpoint3_generator_PUBLIC.py   shared instance generator (already here, don't edit)
src/model/build.py                LPModel, build_b2_model(), build_triangle_model()     -- M1
src/lp/solve.py                   solve_lp()                                            -- M2
src/kkt/checker.py                check_conditions(), check_model()                     -- M3
src/sens/perturb.py               rhs_sensitivity()                                     -- M4
src/bound.py                      gap(), brute_force_binary_optimum()                   -- M5
experiments/run_all.py            regenerates results/*.csv for CP3                     -- M5
tests/test_model.py, test_lp.py, test_kkt.py, test_sens.py, test_bound.py  -- one per role
tests/_pathsetup.py                shared helper (already here, don't edit)
hand/H1_signed.* ... H5_signed.*  each member's own signed hand-task scan
T12_Checkpoint3.tex / .pdf        the submitted CP3 report — added once everyone has pushed
```

## Declarations (AI assistance / external code, Checkpoint 3)

AI assistance: we consulted an AI assistant (Claude) for help while working on this checkpoint.
All code was read, run and verified by the author before committing.

## References

- Lectures 6-7, Nurseitova A.T., Introduction to Optimization (ItO2025). (LP relaxation, duality,
  the KKT/optimality conditions used in Table 2.)
- Q. Huangfu and J. A. J. Hall, "Parallelizing the dual revised simplex method", Mathematical
  Programming Computation, 10(1):119-142, 2018. (The HiGHS solver behind
  `scipy.optimize.linprog(method="highs")`.)
- L. Kleinrock, *Queueing Systems, Volume 1: Theory*, Wiley, 1975. (The M/M/1 waiting-time curve
  $g(u)=u/(1-u)$ that constraint (B4) piecewise-linearizes.)

---

# Checkpoint 4 (Your Own Metaheuristic on the Team's Instance)

Same repo, same instance (`cp34_instance(12, 'B', 2)`, unchanged from Checkpoint 3). Tag for this
checkpoint will be `checkpoint4`. **All individual this time** — no group part; each student
submits their own `Surname_CP4.pdf` to Moodle, and copying code between different
pairs/the solo member is explicitly forbidden (CP4 Instructions, Section 7).

## Who does what (from the team's CP4 task file, final, not to be changed)

| Student | GitHub | Group | Configuration | Role | Partner |
|---|---|---|---|---|---|
| Yermukhamed Tilesbay | [@etilesbay-stack](https://github.com/etilesbay-stack) | SA pair | SA, transfer+swap | SA-I | Aitzhanov Tamerlan |
| Aitzhanov Tamerlan | [@Tabix01](https://github.com/Tabix01) | SA pair | SA, transfer+swap | SA-T | Yermukhamed Tilesbay |
| Zhussupbekova Zere | [@zere90](https://github.com/zere90) | GA pair | GA, penalty | GA-A | Kabdulla Danial |
| Kabdulla Danial | [@Haigatu](https://github.com/Haigatu) | GA pair | GA, penalty | GA-P | Zhussupbekova Zere |
| Noken Rakhima | [@queenbikeshhh](https://github.com/queenbikeshhh) | single | SA, transfer | SA-solo | — |

Reference values (team task file): $z_{LP} = 5.209695$ (matches Checkpoint 3 exactly), $z_{ref} =
5.268322$ (instructor's exact optimum, for reporting only — never read by any `solve(...)`).

## SA-I design (Tilesbay)

- State: `assign`, the server of each VM. Cost: sum over servers of the piecewise-linear latency of the CPU utilisation, updated incrementally (only the two servers touched by a move).
- Moves: transfer (70%) and swap (30%). Zone rule (B2): a VM may only go to a server of its required zone; moves that break it are never generated.
- Constraints: only feasible moves are accepted. A move that overloads CPU or memory is evaluated (it costs one evaluation) and rejected.
- Temperatures: rule of the course (random walk of 3000 proposals accepting everything feasible, T0 = 4.5 · median of positive deltas, Tend = 0.22 · 5th percentile; the walk is charged to the budget), or `params={"T0": ..., "Tend": ...}` from SA-T. Geometric cooling.
- Budget: 100 000 evaluations per run, start solution and every proposal counted, seeds 0-9.

### How to run (SA-I)

```
python check_env.py --instance-seed 12 --story B --variant 2
python tests/test_SA-I.py
python experiments/run_final.py --module src.meta.sa --instance-seed 12 --story B --variant 2 --team T12 --role SA-I --configuration "SA, transfer+swap" --out results/tilesbay_runs.csv
python experiments/analyze_SA-I.py --runs results/tilesbay_runs.csv
```

`analyze_SA-I.py` reruns the ten seeds (deterministic), checks them against the results file and writes Table 1, Table 2, Figure F1 and the acceptance-vs-temperature curve. With tuned temperatures from SA-T add `--params '{"T0": ..., "Tend": ...}'` to both commands. Each test file also runs standalone with `python tests/test_SA-I.py`.

## SA-T design (Tamerlan)

*(Tamerlan: add your part here: Nelder-Mead, tuning experiment, random-search baseline, commands.)*

## GA-A (Zere): how to run

```
pip install numpy matplotlib --break-system-packages   # or inside your venv, no flag needed there
python3 check_env.py --instance-seed 12 --story B --variant 2     # confirm check values first

# GA-A's own role test
python3 tests/test_ga_a.py

# the 10 final runs (seeds 0-9, budget 100000), once src/meta/ga_problem.py (GA-P) is real:
python3 -m experiments.run_final --module src.meta.ga --instance-seed 12 --story B --variant 2 \
    --team T12 --role GA-A --configuration "GA, penalty" --seeds 0-9 --budget 100000 \
    --out results/Zhussupbekova_runs.csv

# the random-search baseline (same budget/seeds, owned by GA-A):
python3 -m experiments.run_final --module src.meta.random_search --instance-seed 12 --story B --variant 2 \
    --team T12 --role GA-A --configuration random --seeds 0-9 --budget 100000 \
    --out results/Zhussupbekova_runs.csv --append

# GA-A's required analysis item: population diversity + effect of N on cost
python3 -m experiments.run_popsize_diversity --instance-seed 12 --story B --variant 2 \
    --N 15 30 --seeds 0-9 --budget 100000 --surname Zhussupbekova
```

`src/meta/ga.py` and `src/meta/random_search.py` were developed and unit-tested against a
local, `.gitignore`d stub (`_dev_ga_problem_stub.py`) implementing the agreed GA-P interface
(`random_individual`, `evaluate`, `crossover`, `mutate`), so the loop, elitism and exact-budget
accounting are verified independently of Danial's real `src/meta/ga_problem.py`. The commands
above will only reproduce real numbers once his file replaces the stub; re-run
`tests/test_ga_a.py` and the final runs again at that point.

## GA-P design (Danial)

*(Danial: add your part here: representation, random_individual, evaluate/penalty formula, crossover, mutate, your three-rho experiment, commands.)*

## SA-solo (Rakhima)

*(Rakhima: add your part here: your sa.py design, the manual tuning grid, commands.)*

## Repository layout (new for Checkpoint 4)

```
checkpoint4_generator_PUBLIC.py   shared instance generator + checkers evaluate_A/evaluate_B
check_env.py                      confirms the check values of the task file
src/meta/interface.py             the solve(...) signature + the checker wrapper (starter kit)
src/meta/sa.py                    solve(), State, moves                                            -- SA-I (Tilesbay)
src/meta/nm.py                    Nelder-Mead from scratch                                          -- SA-T (Tamerlan)
src/meta/ga_problem.py            representation, random_individual, evaluate, crossover, mutate    -- GA-P (Danial)
src/meta/ga.py                    the GA loop: selection, elitism, generations, solve(...)          -- GA-A (Zere)
src/meta/random_search.py         random-search baseline (GA pair)                                  -- GA-A (Zere)
experiments/run_final.py          runs solve on seeds, writes the results file (starter kit)
experiments/analyze_SA-I.py       Table 1, Table 2, F1, acceptance curve                             -- SA-I (Tilesbay)
experiments/tuning.py             T0/Tend tuning experiment                                          -- SA-T (Tamerlan)
experiments/run_popsize_diversity.py  population-diversity / N-size experiment                       -- GA-A (Zere)
tests/test_SA-I.py, test_SA-T.py, test_ga_a.py, test_ga_p.py, test_sa_solo.py   one per role
results/tilesbay_*.csv/.png/.json, Zhussupbekova_*.csv, <surname>_exp.csv, <surname>_tuning.csv   per role/author
```

