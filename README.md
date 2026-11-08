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

Yermukhamed Tilesbay, role SA-I, configuration **SA, transfer+swap**, instance `cp34_instance(12, 'B', 2)`; partner Tamerlan (SA-T). Results: `results/tilesbay_*`.

**Design.**
- State: `assign`, the server of each VM. The cost is the sum over servers of the piecewise-linear latency of the CPU utilisation (the formula of the checker). It is updated incrementally: only the two servers touched by a move change.
- Moves: transfer (70%) and swap (30%, `params["p_swap"]`). Zone rule (B2): a VM may only go to a server of its required zone; moves that break it are never generated.
- Constraint handling: only feasible moves. A move that overloads CPU or memory costs one evaluation and is rejected. No penalty, so the current state is always feasible and `solve` returns the best feasible solution.
- Start: random feasible assignment (most restricted VMs first, each to a random allowed server where it fits); it counts as one evaluation. `start` is accepted and used if given.
- Temperatures: rule of the course. A random walk of 3000 proposals that accepts every feasible move runs inside `solve` and is charged to the budget; `T0 = 4.5 * median(delta > 0)`, `Tend = 0.22 * 5th percentile(delta > 0)`, geometric cooling `T_k = T0 * alpha^k`. With `params={"T0": ..., "Tend": ...}` (from SA-T) there is no walk.
- Budget: 100 000 evaluations per run (start + walk + annealing proposals), seeds 0-9, own `random.Random(seed)`, no global random state, `z_ref` and `z_LP` never read by `solve`.
- Worsening moves in Table 2 and in the acceptance curve: all proposals with Δ > 0, including those rejected for capacity (they are about 7% of the worsening ones; the log keeps a feasibility flag for each).

**Commands** (from the repository root):

```
python check_env.py --instance-seed 12 --story B --variant 2
python tests/test_SA-I.py
python experiments/run_final.py --module src.meta.sa --instance-seed 12 --story B --variant 2 --team T12 --role SA-I --configuration "SA, transfer+swap" --out results/tilesbay_runs.csv
python experiments/run_final.py --module src.meta.sa --instance-seed 12 --story B --variant 2 --team T12 --role SA-I --configuration "SA, transfer+swap (budget check)" --budget 30000 --out results/tilesbay_budget30000.csv
python experiments/analyze_SA-I.py --runs results/tilesbay_runs.csv
```

`analyze_SA-I.py` reruns the ten seeds, stops if a rerun differs from `results/tilesbay_runs.csv`, and writes Table 1, Table 2 and the worst-seed checks to `results/tilesbay_analysis.json`, Figure F1 to `results/F1_SA-I.png` and the acceptance curve to `results/accept_vs_T_SA-I.png`. The prediction written before the final runs is in `results/prediction_SA-I.txt`. With tuned temperatures from SA-T add `--params '{"T0": ..., "Tend": ...}'` to the runner and to the analysis. The report is `Tilesbay_CP4.tex` / `Tilesbay_CP4.pdf`. The `seconds` column depends on the machine.

**Environment.** Python 3.11 or newer; only the standard library, numpy (median and percentile of the walk) and matplotlib (figures) in `src/meta/sa.py` and `experiments/analyze_SA-I.py`. No scipy.optimize, no metaheuristic library. `tests/test_SA-I.py` runs with pytest or standalone.

**SA-I declaration · Tilesbay**

| Tool or source | Used for | Files or sections | What I checked myself |
| --- | --- | --- | --- |
| Claude (AI assistant) | Assisted with implementation suggestions, debugging, code improvements, and experiment analysis. | `src/meta/sa.py`, `tests/test_SA-I.py`, `experiments/analyze_SA-I.py`, `Tilesbay_CP4.tex` | I reviewed the implementation, ran the tests and ten final runs locally, checked the results in `results/tilesbay_runs.csv`, and verified the implementation against the required interface and constraints. |
| Course generator, starter kit, instructions, guide and T12 task file | Instance, checker, runner, protocol, and reference values. | `checkpoint4_generator_PUBLIC.py`, `check_env.py`, `experiments/run_final.py`, `src/meta/interface.py` | `check_env.py` confirmed the task check values: S = 10, M = 35, CPU = 454 and 172, memory = 1680 and 615. |
| NumPy and Matplotlib | Calibration statistics, numerical summaries, and figures. | `src/meta/sa.py`, `experiments/analyze_SA-I.py` | I compared the generated figures and summaries with the values in `results/tilesbay_analysis.json`. |
| Aitzhanov Tamerlan's SA-T results | Temperature parameters and related experiment information. | SA-I temperature configuration and final experiments. | I checked the parameters used in the final runs and verified the resulting costs. |


## SA-T design (Tamerlan)

Configuration: SA, transfer + swap; partner Tilesbay Yermukhamed (SA-I). Instance: `cp34_instance(12, 'B', 2)`.

SA-T owns `src/meta/nm.py`, `experiments/tuning.py`, `experiments/baseline_sa_t.py`, `tests/test_sa_t.py`, `results/Aitzhanov_tuning.csv`, `results/Aitzhanov_tuning_summary.json`, `results/Aitzhanov_NM_trace.png`.

Run from repository root:
```
python check_env.py --instance-seed 12 --story B --variant 2
python -m unittest discover -s tests -p 'test_sa_t.py' -v
python experiments/tuning.py
python experiments/baseline_sa_t.py
python make_report.py
```
`baseline_sa_t.py` appends to tuning CSV; do not execute it twice without regenerating the tuning CSV. Tuning seeds 100–104, inner budget 20,000, 25 Nelder–Mead objective calls. The random baseline evaluates 100,000 independent zone-compatible candidates per seed; infeasible candidates count toward the budget. For final SA, tuning selection is made on tuning seeds only.



**Dependencies and verification.** The SA-T experiments use the SA-I solver implemented by Tilesbay Yermukhamed (`src/meta/sa.py`). Nelder–Mead is implemented separately in `src/meta/nm.py`. The tuning experiments use the SA-I solver through its existing interface without modifying the partner's implementation.

**Prediction status.** The prediction was documented after preliminary exposure to experimental results and is not presented as a pre-registered prediction.

## GA-A (Zere)

| Tool or source | Used for | Files or sections | What you checked yourself |
| --- | --- | --- | --- |
| Course generator, starter kit, instructions, guide and T12 task file | Instance, checker, run protocol, reference values ($z_{ref}$, $z_{LP}$). | GA-A experiments, Tables 1–2, F1 | `check_env.py` matched the task file's check values (S=10, M=35, CPU 172/454, memory 615/1680). |
| NumPy and Matplotlib | Numerical summaries, Figure F1 and the diversity figure. | `ga.py`, `experiments/run_popsize_diversity.py`, analysis | Figures compared against the numbers in the analysis output. |
| Kabdulla Danial's `ga_problem.py` (GA-P, same pair) | The GA operators (representation, `random_individual`, `evaluate`, `crossover`, `mutate`). | Shared GA loop input | I reread his file myself and ran my own `tests/test_ga_a.py` against it (5/5) before trusting it for the final runs. |
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

Kabdulla Danial implements the problem operators for T12 cloud B2. The GA-A loop in `src/meta/ga.py` calls the following functions in `src/meta/ga_problem.py`. All random choices use the NumPy generator passed by the caller. An assignment contains one integer server index per VM.

| Function | Contract |
| --- | --- |
| `random_individual(d, variant, rng)` | Draw a server independently for each VM. A required zone restricts its choices to matching servers. A requirement of −1 allows every server. |
| `evaluate(d, variant, individual)` | Return selection fitness, official checker cost and feasibility. The call to `interface.check` uses `evaluate_B` exactly once. |
| `crossover(parent1, parent2, rng)` | At each VM independently, inherit its server from either parent with probability 0.5. Both parents respect that VM’s zone, so the child does too. |
| `mutate(d, variant, individual, rng)` | Change one or two distinct VMs to a different server in their allowed sets. Preserve the parent. A VM with only one allowed server cannot move. |

**Capacity penalty.** The operators permit overload. For server s, take the positive part of CPU load divided by CPU capacity minus one, and the positive part of memory load divided by memory capacity minus one. Sum both over servers to obtain V. Relative overload makes a 10% excess comparable across resources. Selection minimizes `fitness = cost + rho * V`. For example, CPU 6/4 and memory 10/8 give V = 0.75. The shared final run records the official checker’s cost and feasible flag.

**Penalty experiment.** The prediction and selection rule are in `results/kabdulla_prediction.txt`. Only seeds 100–104 enter tuning. Each value of rho uses 100,000 evaluations per seed, N = 30 and crossover probability 0.9. The offspring denominator excludes the 30 initial individuals and free elite copies. Raw results and aggregates are in `results/kabdulla_exp.csv` and `results/kabdulla_exp_summary.csv`.

| ρ | Feasible offspring | Mean feasible cost | Infeasible runs |
| ---: | ---: | ---: | ---: |
| 1 | 74.92% | 5.296684 | 0/5 |
| 10 | 82.07% | 5.301752 | 0/5 |
| 100 | 87.12% | 5.310238 | 0/5 |

All weights tied on final feasibility. The preregistered mean-cost criterion selected **rho = 1** on the tuning seeds. The higher weight increased the feasible-offspring share by 12.20 percentage points and raised the mean final cost by 0.26%.

**Pair data and worst-seed checks.** Zere published the official pair results in `results/Zhussupbekova_runs.csv`. The GA-P analysis was rerun against those rows. Its ten GA rows and ten random-search rows supply Table 1 and the baseline. The independent GA-P replay produced `results/kabdulla_ga_details.json` with the returned assignments, traces, diversity observations and a snapshot of both starter-runner rows for each seed. The analysis uses this snapshot if the shared GA-A CSV is absent. This preserves the original timings and allows GA-A to track the shared CSV separately. F1 uses the first recorded evaluation at which all ten runs have a feasible incumbent. Table 2 exhausts all single-VM transfers and unordered swaps between VMs on different servers, passes every candidate to the official checker and counts strictly lower cost. The final shortened generation includes its elite. The last full generation is recorded separately. `results/kabdulla_analysis.json` stores Table 1, the worst-seed data, the F1 series, and the move counts.

**Reproduction.** Run these commands from the repository root with NumPy and Matplotlib installed. The example uses a fresh results directory to preserve the original measurements. The recorded run used Python 3.14.3, NumPy 2.4.3 and Matplotlib 3.10.8. Elapsed times depend on the machine and concurrent workers.

```text
python -m unittest tests.test_ga_p -v
python tests/test_ga_a.py
python -m experiments.ga_p_rho --out results/recheck/kabdulla_exp.csv
python -m experiments.ga_p_collect --out-dir results/recheck --workers 5
python -m experiments.ga_p_analysis --results results/recheck
```

To rebuild the tables and figure from the saved runs alone, use `python -m experiments.ga_p_analysis`. The starter runner commands in Zere’s section reproduce the pair CSV directly. The local collector invokes that same runner and observes its first and last populations without drawing extra random numbers. A fresh final run uses the frozen weight in `src/meta/ga_problem.py`.

**GA-P declaration · Kabdulla Danial**

| Tool or source | Used for | Files or sections | What you checked yourself |
| --- | --- | --- | --- |
| OpenAI Codex | Reviewed the existing operators and tests, adjusted experiment bookkeeping, ran the experiments, prepared diagnostics, and drafted the README and report. | GA-P code, results, README and report | Danial chose the pre-final prediction. Codex ran the tests and checker comparisons described above. |
| Course generator, starter kit, instructions, guide and T12 task sheet | Instance, checker, run protocol, reference values and report definitions. | GA-P experiment, Tables 1–2 and F1 | The automated instance check matched 10 servers, 35 VMs, CPU 172/454 and memory 615/1680. Returned assignments were checked again. |
| NumPy and Matplotlib | Seeded variation, numerical summaries and F1. | GA-P operators, role tests and analysis | Automated tests covered zone validity, repeatability, and a hand-computable overload of 0.75. |
| Zere Zhussupbekova’s GA-A code in the same pair | Genetic search loop and random-search baseline. | Final runs and baseline | The original GA-A tests were run with the GA-P operators. Table 1 costs were compared with saved GA solutions and the checker. |

## SA-solo (Rakhima)

Noken Rakhima, role SA-solo, configuration **SA, transfer**, instance `cp34_instance(12, 'B', 2)`. Results: `results/noken_*`.

**File names.** `src/meta/sa.py` belongs to SA-I (Tilesbay), so the solo SA is `src/meta/sa_solo.py`; its `solve(d, variant, seed, budget, params=None, start=None)` has the interface of Section 3. The baseline is my own `src/meta/random_solo.py`; `src/meta/random_search.py` is GA-A's.

**Design.**
- State: `assign`, the server of each VM. The cost is updated incrementally (only the two servers touched by a move change), from a table of the checker's latency values per server and integer CPU load.
- Moves: transfer only. No swap and no variant-specific move. A VM is drawn uniformly among the VMs with at least two allowed servers, the target uniformly among its other allowed servers, so a zone is never broken.
- Constraint handling: only feasible moves. A transfer that overloads CPU or memory costs one evaluation and is rejected. No penalty.
- Start: randomised greedy; the most restricted and largest VMs go first onto a random server that fits. It reads only the data and counts as one evaluation. If no feasible start exists, moves that do not raise the overload are accepted until the state is feasible; a run that never becomes feasible returns `feasible=False`.
- Temperatures: geometric cooling `T_k = T0 * alpha^k`, `alpha = (Tend/T0)^(1/K)`. With `params={"T0": ..., "Tend": ...}` there is no calibration walk and `K = budget - 1`. With `params=None` the 3000-proposal walk of Section 4 runs inside `solve` and is charged to the budget (at most 10% of the budget, for small budgets).
- Worsening moves in Table 2 and the acceptance curve: feasible proposals with Δ > 0. Proposals rejected for capacity have no Δ in this code, are not counted as worsening and are reported separately.

**Manual tuning grid** (role table, seeds 100-104, budget 100 000): `T0` and `Tend` are the rule-based values (walk with seed 100: `T0 = 0.870182`, `Tend = 0.003059`) times 1/3, 1 or 3, so 2 parameters with 3 values each. The lowest mean cost is kept, and the rule-based cell wins a tie. Selected: `T0 = 0.29006072102002334`, `Tend = 0.0010197932600936345`. The final runs use these values, so `K = 99 999`.

**Commands** (from the repository root, Python 3.9.6, NumPy 2.0.2, Matplotlib 3.9.4):

```
python check_env.py --instance-seed 12 --story B --variant 2
python tests/test_sa_solo.py
python experiments/tune_sa_solo.py
python experiments/run_final.py --module src.meta.sa_solo --instance-seed 12 --story B --variant 2 --team T12 --role SA-solo --configuration "SA, transfer" --seeds 0-9 --budget 100000 --params '{"T0": 0.29006072102002334, "Tend": 0.0010197932600936345}' --out results/noken_runs.csv
python experiments/run_final.py --module src.meta.random_solo --instance-seed 12 --story B --variant 2 --team T12 --role SA-solo --configuration random --seeds 0-9 --budget 100000 --out results/noken_runs.csv --append
python experiments/analyze_sa_solo.py
```

`analyze_sa_solo.py` reruns the ten seeds, stops if a rerun differs from `results/noken_runs.csv`, and writes Table 1, Table 2 and the worst-seed checks to `results/noken_analysis.json`, Figure F1 to `results/F1_noken.png` and the acceptance curve to `results/accept_vs_T_noken.png`. The tuning grid is in `results/noken_tuning.csv` and `results/noken_tuning_summary.csv`; the selected parameters are in `results/noken_params.json`. Rows in `results/noken_runs.csv` with `configuration` = `random` are the baseline. The `seconds` column depends on the machine.

**Declaration · Noken Rakhima**

| Tool or source | Used for | Files or sections | What was checked |
| --- | --- | --- | --- |
| Author's work & custom scripts | Implemented the SA, the baseline, the role test, the tuning and analysis scripts and this README section; ran the experiments; prepared the prediction before the final runs; wrote the report text and designed the PDF layout. | `src/meta/sa_solo.py`, `src/meta/random_solo.py`, `tests/test_sa_solo.py`, `experiments/tune_sa_solo.py`, `experiments/analyze_sa_solo.py`, `results/noken_prediction.txt`, this section, `Noken_CP4.pdf` | The 11 checks of `tests/test_sa_solo.py` pass (incremental cost against the checker on 1000 moves up to 1e-9, same seed, exact budget, monotone trace, zones, honest feasible flag). The analysis reruns all ten seeds against the results file; the moves of Table 2 are judged by the checker. |
| Course generator, starter kit, instructions, guide and T12 task file | Instance, checker, runner, protocol, reference values. | `checkpoint4_generator_PUBLIC.py`, `check_env.py`, `experiments/run_final.py`, `src/meta/interface.py` | `check_env.py` prints the check values of the task file: S 10, M 35, CPU 454 and 172, memory 1680 and 615. |
| NumPy and Matplotlib | Median and percentile of the calibration walk, summaries, figures. | `src/meta/sa_solo.py`, `experiments/analyze_sa_solo.py` | Figures were compared with the numbers in `results/noken_analysis.json`. |


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

