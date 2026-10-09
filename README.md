# T12 — Checkpoint 2 (Unconstrained Component)

Team T12 · Storyline B (cloud resource allocation) · variant 2 · seed 12

## Roles

Chosen for a team of five, per Section 5 of the instructions (role table).
**Fill in real names before the first commit** — the placeholders below
must become the actual GitHub-linked names before anything is pushed.

| Role | Member | Code authored (`src/`) | Analysis section | Hand trace |
|------|--------|-------------------------|-------------------|------------|
| M1 — GD | Tilesbay | `optim/gd.py`, `problems/rosenbrock.py`, `check_grad` | S1 — Step size and stability | H1 |
| M2 — Newton | Rakhima | `optim/newton.py`, `problems/quadratic.py` | S2 — Conditioning, cost and safeguards | H2 |
| M3 — Momentum | Tamerlan | `optim/momentum.py`, `experiments/plots.py` | S3 — Momentum in the valley | H3 |
| M4 — Adam | Danial | `optim/adam.py` | S4 — Adam and the rotated axes | H4 |
| M5 — Project lead | Zere | `problems/project.py`, `experiments/run_all.py` (tables), assembling the PDF | S5 — Project block | H5 |

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

## Declarations (AI assistance / external code)

**AI assistance.** This checkpoint's code (`src/optim/`, `src/problems/`,
`src/experiments/`), tests, figures, hand-trace reference derivations,
and the LaTeX report were drafted with the help of an AI assistant
(Claude), at the request of one team member, working from the course's
own `Checkpoint_2_Instructions.docx`, team task file, and
`checkpoint2_generator_PUBLIC.py`. This is declared here per Section 2
("Cite and declare... code produced with an AI assistant must be
declared in the Declarations section of README.md").

**What this means for grading, concretely — please read before you push
anything:**

1. **I1 (6 pts, individual, "visible authored commit history") is not
   satisfiable by importing this repository and pushing it as one
   commit.** The rubric explicitly keys individual credit to Git
   authorship ("the author field decides") and imposes an *automatic
   zero* on a member whose contribution "cannot be attributed in the Git
   history." Each of the five of you needs your **own** commits, under
   **your own** `git config user.name`/`user.email` (linked to your
   GitHub account), that show you actually working on your own file —
   not one person pasting the whole tree in. The honest way to use this
   draft is: each member opens *their own* file (e.g. the GD owner opens
   only `src/optim/gd.py` and `src/problems/rosenbrock.py`), reads it,
   runs it, tweaks/re-derives something themselves (a comment, a variable
   name, a check, a docstring correction — whatever makes it genuinely
   theirs), and commits *that* under their own identity, tagged
   `[GD]`/`[Newton]`/`[Momentum]`/`[Adam]`/`[Infra]`/`[Report]` as
   Section 7.2 requires. Spread this over more than one sitting if you
   can — "one giant commit the night before does not show an
   implementation history."

2. **I2 (5 pts, hand traces) explicitly requires the trace to be done on
   paper and signed by the person being graded.** The `hand/` folder in
   this repository contains *reference derivations* with the correct
   numbers (verified against the real solver code via the `tests/`
   files) — not something to print and submit as-is. Each role-owner
   must redo their own H-number by hand, check it against the reference,
   and sign their own page. A hand trace a student did not actually work
   through is exactly what I2 is designed to catch and reward, so doing
   this for real is in each member's own interest, not just a rule.

3. **Everything in this declaration is itself required** — Section 2
   says undeclared AI/external-code use "falls under the academic
   integrity policy in the syllabus." This section exists so that use is
   declared, not hidden.

4. **The Git repository itself (creation, collaborator invitations,
   the instructor's invite to `@altynbatyn`, the per-member pushes, the
   `checkpoint2` tag) has to be done by the team, by hand, with real
   GitHub accounts** — none of that can come from this draft. This
   repository tree is meant to be copied into a private repo one of you
   creates, then built on top of with real individual commits as in (1).

**External code.** None beyond the standard library and `numpy`
(`src/`) and `matplotlib` (`experiments/plots.py` only), as permitted by
Section 2.
