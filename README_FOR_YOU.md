# STEP 0 — Zere: push this BEFORE anyone else starts their Checkpoint 3 work

This is not a new repo — it drops straight into the **existing** `12-checkpoint2` repo on GitHub
(the one from Checkpoint 2). Nothing in here overwrites any Checkpoint 2 file.

## What's in this zip
```
README.md                         REPLACES your current README.md — keeps the whole Checkpoint 2
                                   section unchanged, adds a new "Checkpoint 3" section below it
checkpoint3_generator_PUBLIC.py   new file at the repo root
tests/_pathsetup.py               new shared test helper (tests/__init__.py already exists, untouched)
src/model/__init__.py             new, empty (M1 fills in build.py next)
src/lp/__init__.py                new, empty (M2 fills in solve.py next)
src/kkt/__init__.py               new, empty (M3 fills in checker.py next)
src/sens/__init__.py              new, empty (M4 fills in sens/perturb.py next)
experiments/__init__.py           new, empty, at the repo ROOT (separate from src/experiments/,
                                   which stays exactly as Checkpoint 2 left it)
```

## Do this, in order

1. Open your existing local clone: `cd ~/PycharmProjects/12-checkpoint2` (or wherever you cloned
   it — the same folder you used for Checkpoint 2).
2. `git pull` first, to make sure you're starting from the latest `main`.
3. Copy every file from this zip into that folder, **merging** with what's already there (don't
   delete anything existing — this only adds new files/folders and replaces `README.md`).
4. Check nothing got clobbered:
   ```
   git status
   ```
   You should see new files (`checkpoint3_generator_PUBLIC.py`, `src/model/`, `src/lp/`,
   `src/kkt/`, `src/sens/`, `experiments/__init__.py`, `tests/_pathsetup.py`) and `README.md`
   modified. Nothing under `src/optim/`, `src/problems/`, `src/experiments/`, or `results/` should
   show as changed.
5. Commit and push:
   ```
   git add README.md checkpoint3_generator_PUBLIC.py tests/_pathsetup.py \
           src/model/__init__.py src/lp/__init__.py src/kkt/__init__.py src/sens/__init__.py \
           experiments/__init__.py
   git commit -m "[Report] checkpoint 3 skeleton: structure, shared generator, roles"
   git push
   ```
6. Tell the other four to `git pull` before they start — then send them their own zip (M1–M4
   below) and have them follow their own `README_FOR_YOU.md`.
