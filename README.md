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

**AI assistance.** A working draft of this checkpoint's code (`src/model/`, `src/lp/`, `src/kkt/`,
`src/sens/`, `src/bound.py`, `experiments/run_all.py`), tests, the hand-task reference derivations,
and the LaTeX report were drafted with the help of an AI assistant (Claude), at the request of one
team member, working from `Checkpoint_3_Instructions.docx`, our team task file, and
`checkpoint3_generator_PUBLIC.py`. Declared here per Section 4.

**What this means for grading — please read before pushing:**

1. **I1 (5 pts, "correct, authored in Git, the test reproduces your hand numbers") is not
   satisfiable by importing the draft and pushing it as one commit.** Section 8 keys individual
   credit to Git authorship, with automatic zero for a member whose contribution can't be
   attributed. Each member commits **their own** file under their **own** `git config
   user.name`/`user.email`, after reading it, running it, and changing something real to make it
   genuinely theirs — tagged `[Model]`/`[LP]`/`[KKT]`/`[Sens]`/`[Report]`.
2. **I3 (4 pts, hand task) requires a trace actually worked through on paper and signed.** The
   `hand/H#_reference.md` files are *reference* derivations to check your own work against, not to
   copy. Do your own derivation first, then check it, then sign your own scan.
3. **External code:** none beyond `numpy` and `scipy` (`src/lp` only), as permitted by Section 4.
